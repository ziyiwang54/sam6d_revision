#!/usr/bin/env python
import os
import sys
import logging
import json

import numpy as np
import torch
import trimesh
from PIL import Image
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
import pycocotools.mask as cocomask
from torchvision import transforms
from types import SimpleNamespace
from omegaconf import OmegaConf

# ─── USER CONFIGURATION ─────────────────────────────────────────────────────────
OUTPUT_DIR         = "output/"
TEMPLATE_DIR       = "Data/Example/outputs/templates"  
CAD_PATH           = "Data/Example/mesh/obj_landingpole.ply"
RGB_PATH           = "Data/Example/rgb/rgb_landingpole_mid.png"
DEPTH_PATH         = "Data/Example/depth/depth_landingpole_mid.png"
CAM_PATH           = "Data/Example/camera_intrinsics/camera.json"

FASTSAM_CHECKPT    = os.path.join(
    os.path.dirname(__file__),
    "Instance_Segmentation_Model","checkpoints","FastSAM","FastSAM-x.pt"
)
FASTSAM_CFG        = {"iou_threshold":0.30, "conf_threshold":0.50, "max_det":50}
FASTSAM_WIDTH      = 1024

PEM_CHECKPT        = os.path.join(
    os.path.dirname(__file__),
    "Pose_Estimation_Model","checkpoints","sam-6d-pem-base.pth"
)
DET_SCORE_THRESH   = 0.2
TOP_K              = 5

# IMG_SIZE                = 224
# N_SAMPLE_OBS_POINT      = 8192
# N_SAMPLE_MODEL_POINT    = 10000
RGB_MASK_FLAG           = False

# ─── BOILERPLATE ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PEM_CONFIG_PATH = os.path.join(BASE_DIR, "Pose_Estimation_Model", "config","base.yaml")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Instance_Segmentation_Model"))
pem_root = os.path.join(BASE_DIR, "Pose_Estimation_Model")
sys.path.insert(0, os.path.join(pem_root, "provider"))
sys.path.insert(0, os.path.join(pem_root, "utils"))
sys.path.insert(0, os.path.join(pem_root, "model"))
sys.path.insert(0, os.path.join(pem_root, "model", "pointnet2"))

# ─── IMPORTS ────────────────────────────────────────────────────────────────────
from segment_anything.utils.amg import rle_to_mask
from Instance_Segmentation_Model.model.fast_sam import FastSAM
from Instance_Segmentation_Model.utils.poses.pose_utils import (
    get_obj_poses_from_template_level,
    load_index_level_in_level2,
)
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23

# **Always these two**:
from Pose_Estimation_Model.utils.data_utils import (
    load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose,
)
from Pose_Estimation_Model.utils.draw_utils import draw_detections
from Pose_Estimation_Model.model.pose_estimation_model import Net as PEMNet

inv_rgb_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229, 1/0.224, 1/0.225],
    ),
])
rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ─── VISUALIZATION HELPERS ──────────────────────────────────────────────────────
def visualize_seg(rgb_pil, dets, out_file):
    img     = np.array(rgb_pil)
    gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors  = distinctipy.get_colors(len(dets)); α = 0.33
    best    = max(dets, key=lambda d: d["score"])
    msk     = rle_to_mask(best["segmentation"])
    edge    = binary_dilation(canny(msk), np.ones((2,2)))
    cid     = best["category_id"] - 1
    r,g,b   = [int(255*c) for c in colors[cid]]
    overlay[msk,0] = α*r + (1-α)*overlay[msk,0]
    overlay[msk,1] = α*g + (1-α)*overlay[msk,1]
    overlay[msk,2] = α*b + (1-α)*overlay[msk,2]
    overlay[edge,:] = 255
    vis = Image.fromarray(np.uint8(overlay))
    out = Image.new("RGB",(rgb_pil.width+vis.width, rgb_pil.height))
    out.paste(rgb_pil,(0,0)); out.paste(vis,(rgb_pil.width,0))
    out.save(out_file)
    return out

# def visualize_pose(rgb_np, R, t, model_pts, K, out_file):
#     pose_img = draw_detections(rgb_np, R, t, model_pts, K, color=(255,0,0))
#     ppil     = Image.fromarray(np.uint8(pose_img))
#     rpil     = Image.fromarray(np.uint8(rgb_np))
#     out = Image.new("RGB",(rgb_np.shape[1]+ppil.width, rgb_np.shape[0]))
#     out.paste(rpil,(0,0)); out.paste(ppil,(rgb_np.shape[1],0))
#     out.save(out_file)
#     return out
def visualize_pose(rgb_np, R, t, model_pts, K, out_file):
    # ensure batch dimensions for rotations, translations, and intrinsics
    if R.ndim == 2:
        R_batch = R[np.newaxis, ...]
    else:
        R_batch = R

    if t.ndim == 1:
        t_batch = t[np.newaxis, ...]
    else:
        t_batch = t

    if K.ndim == 2:
        K_batch = K[np.newaxis, ...]
    else:
        K_batch = K

    # draw_detections expects batched inputs
    pose_img = draw_detections(
        rgb_np,
        R_batch,
        t_batch,
        model_pts,
        K_batch,
        color=(255, 0, 0),
    )
    ppil = Image.fromarray(np.uint8(pose_img))
    rpil = Image.fromarray(np.uint8(rgb_np))

    # stitch original and pose image side by side
    out = Image.new(
        "RGB",
        (rgb_np.shape[1] + ppil.width, max(rgb_np.shape[0], ppil.height))
    )
    out.paste(rpil, (0, 0))
    out.paste(ppil, (rgb_np.shape[1], 0))
    out.save(out_file)
    return out


# ─── SEGMENTATION (ISM) ─────────────────────────────────────────────────────────
def run_segmentation():
    logging.info("→ Initializing FastSAM")
    cfg = SimpleNamespace(**FASTSAM_CFG)
    sam = FastSAM(
        checkpoint_path      = FASTSAM_CHECKPT,
        config               = cfg,
        segmentor_width_size = FASTSAM_WIDTH,
        device               = device,
    )

    # load + preprocess
    rgb_pil = Image.open(RGB_PATH).convert("RGB")
    rgb_np  = np.array(rgb_pil)

    # **PASS array** to get back a dict of masks+boxes
    mask_data = sam.generate_masks(rgb_np)
    dets      = Detections(mask_data)

    # move to CPU for any future numpy ops
    dets.boxes = dets.boxes.cpu()
    dets.masks = dets.masks.cpu()

    # convert back to COCO‐style list-of-dicts for the next step
    coco_dets = []
    N = dets.masks.shape[0]
    for i in range(N):
        mask_np = dets.masks[i].numpy().astype(np.uint8)
        rle = cocomask.encode(np.asfortranarray(mask_np))
        coco_dets.append({
            "segmentation": rle,
            "score":        float(dets.scores[i]) if hasattr(dets, "scores") else 1.0,
            "category_id":  int(dets.labels[i]) if hasattr(dets, "labels") else 1,
        })

    # visualize
    vis_path = os.path.join(OUTPUT_DIR, "sam6d_results", "vis_ism.png")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    visualize_seg(rgb_pil, coco_dets, vis_path)

    return coco_dets, rgb_np

# ─── POSE‐ESTIMATION (PEM) ──────────────────────────────────────────────────────
def get_test_data(raw_js, ds_cfg):
    dets = sorted(
        [d for d in raw_js if d["score"] > DET_SCORE_THRESH],
        key=lambda d: d["score"], reverse=True
    )[:TOP_K]
    if not dets:
        raise RuntimeError("No ISM masks above threshold")

    cam = json.load(open(CAM_PATH))
    K   = np.array(cam["cam_K"]).reshape(3,3)

    img = load_im(RGB_PATH)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=2)
    img = img.astype(np.uint8)

    depth = load_im(DEPTH_PATH).astype(np.float32) * cam["depth_scale"] / 1000.0
    pts3  = get_point_cloud_from_depth(depth, K)

    mesh      = trimesh.load_mesh(CAD_PATH)
    model_pts = mesh.sample(ds_cfg.n_sample_model_point).astype(np.float32) / 1000.0
    rad       = np.max(np.linalg.norm(model_pts, axis=1))

    all_pts, all_rgb, all_choose, scores, kept = [],[],[],[],[]
    for d in dets:
        seg = d["segmentation"]; h,w = seg["size"]
        try: rle = cocomask.frPyObjects(seg, h, w)
        except: rle = seg
        mask = cocomask.decode(rle) & (depth > 0)
        if mask.sum() < 32: continue

        y1,y2,x1,x2 = get_bbox(mask)
        m           = mask[y1:y2, x1:x2]
        choose      = m.flatten().nonzero()[0]

        cloud = pts3[y1:y2, x1:x2, :].reshape(-1,3)[choose]
        c0    = cloud.mean(0)
        keep_mask = np.linalg.norm(cloud-c0, axis=1) < rad*1.2
        if keep_mask.sum() < 4: continue

        total = keep_mask.sum()
        idxs = np.random.choice(
            total,
            ds_cfg.n_sample_observed_point,
            replace=(total < ds_cfg.n_sample_observed_point),
        )
        choose = choose[keep_mask][idxs]
        cloud  = cloud[keep_mask][idxs]

        patch = img[y1:y2, x1:x2][:,:,::-1]
        patch = cv2.resize(patch, (ds_cfg.img_size, ds_cfg.img_size), interpolation=cv2.INTER_LINEAR)
        if RGB_MASK_FLAG:
            patch = patch * (m[:,:,None] > 0).astype(np.uint8)
        patch_t = rgb_transform(patch)
        rc      = get_resize_rgb_choose(choose, [y1,y2,x1,x2], ds_cfg.img_size)

        all_pts.append(torch.FloatTensor(cloud))
        all_rgb.append(torch.FloatTensor(patch_t))
        all_choose.append(torch.LongTensor(rc))
        scores.append(d["score"])
        kept.append(d)

    n = len(all_pts)
    data = {
        "pts":        torch.stack(all_pts).to(device),
        "rgb":        torch.stack(all_rgb).to(device),
        "rgb_choose": torch.stack(all_choose).to(device),
        "score":      torch.FloatTensor(scores).to(device),
        "model":      torch.FloatTensor(model_pts).unsqueeze(0).repeat(n,1,1).to(device),
        "K":          torch.FloatTensor(K).unsqueeze(0).repeat(n,1,1).to(device),
    }
    return data, kept, model_pts

def get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz

def get_templates(path, cfg):
    """
    path: folder with rgb_i.png, mask_i.png, xyz_i.npy
    cfg: the OmegaConf section (cfg.test_dataset) containing
         n_template_view, img_size, n_sample_template_point, rgb_mask_flag
    """
    total_views = 42
    all_tem, all_tem_pts, all_tem_choose = [], [], []

    for v in range(cfg.n_template_view):
        idx = int(total_views / cfg.n_template_view * v)
        tem, tem_choose, tem_pts = get_template(path, cfg, idx)
        # add batch dim and move to device
        all_tem.append(    torch.FloatTensor(tem).unsqueeze(0).to(device))
        all_tem_choose.append(torch.LongTensor(tem_choose).unsqueeze(0).to(device))
        all_tem_pts.append( torch.FloatTensor(tem_pts).unsqueeze(0).to(device))
    return all_tem, all_tem_pts, all_tem_choose

def run_pose(raw_js, rgb_np):
    logging.info("→ Initializing PEM model")
    cfg = OmegaConf.load(PEM_CONFIG_PATH)
    # instantiate PEMNet exactly like the original code did:
    pem = PEMNet(cfg.model).to(device).eval()
    ckpt  = torch.load(PEM_CHECKPT, map_location=device)
    # pem.load_state_dict(ck.get("state_dict", ck))
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    pem.load_state_dict(state_dict, strict=False)

    data, kept, model_pts = get_test_data(raw_js, cfg.test_dataset)
    n = data["pts"].size(0)

    # 3) --- NEW TEMPLATE PREP STEPS ---
    # this returns lists of (rgb, xyz, choose) tensors
    all_tem, all_tem_pts, all_tem_choose = get_templates(TEMPLATE_DIR, cfg.test_dataset)  
    # now extract features from those templates
    with torch.no_grad():
        # get_obj_feats returns (pts_feats, rgb_feats)
        all_tem_pts, all_tem_feat = pem.feature_extraction.get_obj_feats(
            all_tem, all_tem_pts, all_tem_choose
        )

    with torch.no_grad():
        data["dense_po"] = all_tem_pts.repeat(n,1,1).to(device)
        data["dense_fo"] = all_tem_feat.repeat(n,1,1).to(device)
        out = pem(data)

    if "pred_pose_score" in out:
        pose_scores = (out["pred_pose_score"] * out["score"]).cpu().numpy()
    else:
        pose_scores = out["score"].cpu().numpy()
    R = out["pred_R"].cpu().numpy()
    t = out["pred_t"].cpu().numpy() * 1000

    res_dir = os.path.join(OUTPUT_DIR, "sam6d_results")
    with open(os.path.join(res_dir, "detection_pem.json"), "w") as f:
        # for i,d in enumerate(kept):
        #     d["score"], d["R"], d["t"] = float(pose_scores[i]), R[i].tolist(), t[i].tolist()
        # json.dump(kept, f)
        for i, d in enumerate(kept):
            # update score, rotation & translation
            d["score"] = float(pose_scores[i])
            d["R"]     = R[i].tolist()
            d["t"]     = t[i].tolist()
            # if segmentation is RLE, its 'counts' may be bytes—decode to str
            seg = d.get("segmentation", {})
            cnt = seg.get("counts", None)
            if isinstance(cnt, (bytes, bytearray)):
               seg["counts"] = cnt.decode("ascii")
        # now safe to serialize
        json.dump(kept, f)

    best = pose_scores.argmax()
    visualize_pose(
        rgb_np,
        R[best], t[best],
        model_pts * 1000,
        data["K"][best].cpu().numpy(),
        os.path.join(res_dir, "vis_pem.png"),
    )

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_DIR, "sam6d_results"), exist_ok=True)

    seg_js, rgb_np = run_segmentation()
    run_pose(seg_js, rgb_np)

    im1 = Image.open(os.path.join(OUTPUT_DIR, "sam6d_results", "vis_ism.png"))
    im2 = Image.open(os.path.join(OUTPUT_DIR, "sam6d_results", "vis_pem.png"))
    combo = Image.new(
        "RGB",
        (im1.width + im2.width, max(im1.height, im2.height))
    )
    combo.paste(im1, (0,0))
    combo.paste(im2, (im1.width,0))
    combo.save(os.path.join(OUTPUT_DIR, "sam6d_results", "vis_combined.png"))
