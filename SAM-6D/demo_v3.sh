# Render CAD templates
export OUTPUT_DIR=$(realpath -q Data/Example/outputs/)
# export CAD_PATH=$(realpath -q Data/Example/mesh/obj_stop*.ply)
# export CAD_PATH=$(realpath -q Data/Example/mesh/obj_connector.ply)
# export CAD_PATH=$(realpath -q Data/Example/mesh/connection.ply)
export CAD_PATH=$(realpath -q Data/Example/mesh/obj_landingpole.ply)
# export CAD_PATH=$(realpath -q Data/Example/mesh/obj_000005.ply)
# export RGB_PATH=$(realpath -q Data/Example/rgb/rgb.png )
# export DEPTH_PATH=$(realpath -q Data/Example/depth/depth.png)
# export RGB_PATH=$(realpath -q Data/Example/rgb/rgb_stop*.png )
# export DEPTH_PATH=$(realpath -q Data/Example/depth/depth_stop*.png)
# export RGB_PATH=$(realpath -q Data/Example/rgb/rgb_connector_close_scene.png)
# export DEPTH_PATH=$(realpath -q Data/Example/depth/depth_connector_close_scene.png)
export RGB_PATH=$(realpath -q Data/Example/rgb/rgb_landingpole_mid.png)
export DEPTH_PATH=$(realpath -q Data/Example/depth/depth_landingpole_mid.png)
export CAMERA_PATH=$(realpath -q Data/Example/camera_intrinsics/camera.json)

export RENDER_DIR=$(realpath -q Render)
export ISM_DIR=$(realpath -q Instance_Segmentation_Model)
export PEM_DIR=$(realpath -q Pose_Estimation_Model)

if [ "$1" = "render" ]; then
    export RENDER_FLAG="render"
else
    export RENDER_FLAG="no_render"
fi
if [ $RENDER_FLAG = "render" ]; then
    rm -rf $OUTPUT_DIR/templates
    cd $RENDER_DIR
    blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 
fi
# # Run instance segmentation model
export SEGMENTOR_MODEL=fastsam

cd $ISM_DIR
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH


# # Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd $PEM_DIR
python run_inference_custom_v3.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH # --debug True

cd $OUTPUT_DIR/sam6d_results
python visualize.py