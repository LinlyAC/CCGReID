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
Download the CCGReID dataset and modify the dataset path.
Line 23 and 73 in  [prcc.py](./fastreid/data/datasets/prcc.py) 
> dataset_dir = XXX


Maybe you should also keep the same dataset folder name in line 31.

Same operations in [vc.py](./fastreid/data/datasets/vc.py).


#### Prepare ViT Pre-trained Models
Download the ViT Pre-trained model and modify the path, line 11 in [SSRG.yml](./configs/Base-SSRG.yml):
> PRETRAIN_PATH: XXX


