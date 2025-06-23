### Create conda environment
# conda env create -f environment.yaml
# conda activate sam6d

# ### Install pointnet2
# cd Pose_Estimation_Model/model/pointnet2
# python setup.py install
# cd ../../../

### Download ISM pretrained model
export ISM_DIR=$(realpath -q Instance_Segmentation_Model)
export PEM_DIR=$(realpath -q Pose_Estimation_Model)

cd $ISM_DIR
# python download_sam.py
python download_fastsam.py
python download_dinov2.py

### Download PEM pretrained model
cd $PEM_DIR
python download_sam6d-pem.py
