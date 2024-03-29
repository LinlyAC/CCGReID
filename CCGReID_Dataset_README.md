# CCGReID Datasets
Two datasets for cloth-changing group re-identification (CCGReID) task, namely **GroupPRCC** and **GroupVC** dataset.

## Introduction
* The GroupPRCC dataset contains 7,514 training images of 64 group identities and 3,676 testing images of 32 group identities, which are captured under 3 cameras.
* The GroupVC dataset contains 3,832 training images of 111 group identities and 4,179 testing images of 118 group identities, which are captured under 4 cameras.
* Both datasets provide SCS (same-cloth setting) and CCS (cross-cloth setting) protocols, corresponding x_SCS.txt and x_CCS.txt, where x $\in${query, gallery}.

Please note: For ease of storage and distribution, we store the member (size: $256\times128$) in each group in a single, uniform-sized image (size: $256\times768$) with proper zero padding. The image of each member can be easily obtained by dividing the group image into six equal parts and discarding obvious zero padding.

## Annotation
The annotation in both datasets has the same format:
```
Img_Path Group_ID,Camera_ID Num_Of_Total_Member,Num_Of_CC_Member ClothID_Of_CC_Member
```
For example, the label ` ./train/0_1_2.jpg 0,1 5,2 1,1 ` means the group ID and camera ID of the image (./train/0_1_2.jpg) are 0 and 1. This image contains 5 members and there are 2 cloth-changing members within them. (Optionally, the cloth IDs of current cloth-changing members are also provided, as 1 and 1, respectively.)

Note: Some operations that may reveal the identity of cloth-changing members are not recommended.

## Download Link
* GroupPRCC dataset [Google Drive](https://drive.google.com/file/d/1m4O_G3Bdl9IBEYsLCJJnwLQLS8t3CtSR/view?usp=drive_link)
* GroupVC dataset [Google Drive](https://drive.google.com/file/d/1f0YFpND6iQkENabiD0-DR0LQQSp2HRxA/view?usp=drive_link)

## References
If you find our datasets useful for your research, please kindly cite the following papers.

```
@ARTICLE{GCCReID,
  title={Separable Spatial-Temporal Residual Graph for Cloth-Changing Group Re-Identification}, 
  author={Zhang, Quan and Lai, Jianhuang and Xie, Xiaohua and Jin, Xiaofeng and Huang, sien},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2024}
}

@inproceedings{City1M_dataset,
  title={Modeling 3D Layout for Group Re-Identification},
  author={Zhang, Quan and Dang, Kaiheng and Lai, Jian-Huang and Feng, Zhanxiang and Xie, Xiaohua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022},
  month={June},
  pages={7512-7520}
}
```

## License
* The datasets can only be used for ACADEMIC PURPOSES. NO COMERCIAL USE is allowed.
* Copyright © Sun Yat-sen University. All rights reserved.

## Contact
If you have any questions, please feel free to contact us. E-mail: zhangq48@mail2.sysu.edu.cn

