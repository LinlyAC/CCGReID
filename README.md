# Cloth-changing Group Re-identification (CCGReID)
Implementation of our paper,  Separable Spatial-Temporal Residual Graph for Cloth-Changing Group Re-Identification (TPAMI'24), including dataset (GroupPRCC and GroupVC) and method (SSRG).

## CCGReID Datasets_part
Please refer to [dataset_readme.md](./CCGReID_Dataset_README.md).

## CCGReID Method_part
Please refer to [dataset_readme.md](./CCGReID_Dataset_README.md)

### Requirements
#### Step1: Prepare enviorments
First refer to [INSTALL.md](./INSTALL.md).
After that, the **PyTorch Geometric** package is also needed.

#### Step1: Prepare datasets
Download the CCGReID dataset and modify the dataset path, line 26 in [csg.py](./fastreid/data/datasets/CSG.py) (./fastreid/data/datasets/CSG.py):
> self.root = XXX

#### Prepare ViT Pre-trained Models
Download the ViT Pre-trained model and modify the path, line 11 in [bagtricks_gvit.yml](./configs/CSG/bagtricks_gvit.yml) (./configs/CSG/bagtricks_gvit.yml):
> PRETRAIN_PATH: XXX


