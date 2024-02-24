# Cloth-changing Group Re-identification (CCGReID)
Implementation of our paper,  **Separable Spatial-Temporal Residual Graph for Cloth-Changing Group Re-Identification (TPAMI'24)**, including dataset (GroupPRCC and GroupVC) and method (SSRG).

## CCGReID Datasets_part
Please refer to [CCGReID_Dataset_README.md](./CCGReID_Dataset_README.md).

## CCGReID Method_part
### Requirements
#### Step1: Prepare enviorments
First refer to [INSTALL.md](./INSTALL.md).
After that, the **PyTorch Geometric** package is also needed.

#### Step2: Prepare datasets
Download the CCGReID dataset and modify the dataset path.
Line 23 and 73 in  [prcc.py](./fastreid/data/datasets/prcc.py) .
> dataset_dir = XXX


Maybe you should also keep the same dataset folder name in line 31.

Same operations in [vc.py](./fastreid/data/datasets/vc.py).

#### Step3: Prepare ViT Pre-trained Models
Download the ViT Pre-trained model and modify the path, line 11 in [SSRG.yml](./configs/Base-SSRG.yml):
> PRETRAIN_PATH: XXX

### Training
Single or multiple GPU training is supported. Please refer to [scripts](./scripts/) folder.

## Acknowledgement
Codebase from [fast-reid](https://github.com/JDAI-CV/fast-reid). So please refer to that repository for more usage.

## Citation
If you find this code useful for your research, please kindly cite the following papers:
```
@ARTICLE{10443971,
  author={Zhang, Quan and Lai, Jianhuang and Xie, Xiaohua and Jin, Xiaofeng and Huang, Sien},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Separable Spatial-Temporal Residual Graph for Cloth-Changing Group Re-Identification}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TPAMI.2024.3369483}}

@InProceedings{Zhang_2022_CVPR,
    author    = {Zhang, Quan and Dang, Kaiheng and Lai, Jian-Huang and Feng, Zhanxiang and Xie, Xiaohua},
    title     = {Modeling 3D Layout for Group Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    month     = {June},
    year      = {2022},
    pages     = {7512-7520}
}
```

## Contact
If you have any question, please feel free to contact me. E-mail: zhangq48@mail2.sysu.edu.cn


