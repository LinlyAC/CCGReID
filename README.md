# CCGReID Datasets
Cloth-changing group re-identification dataset, incluidng **GroupPRCC** and **GroupVC** datasets.

## Introduction
* GroupPRCC dataset contains 7,514 training images of 64 group identities and 3,676 testing images of 32 group identities, which are captured under 3 cameras.
* GroupVC dataset contains 3,832 training images of 111 group identities and 4,179 testing images of 118 group identities, which are captured under 4 cameras.
* Both datasets provide SCS (same-cloth setting) and CCS (cross-cloth setting) protocols, corrsponding x_SCS.txt and x_CCS.txt, where x $\in${query,gallery}.

Please note: For ease of storage and distribution, we store the member (size: $256\times128$) in each group in a single, uniform-sized image (size: $256\times768$) with proper zero padding. The image of each member can be easily obtained by dividing the group image in six equal parts and discarding obvious zero padding.

## Annotation
The annotation in both datasets have the same format:
```
Img_Path Group_ID,Camera_ID Num_Of_Total_Member,Num_Of_CC_Member ClothID_Of_CC_Member
```
For example, label ` ./train/0_1_2.jpg 0,1 5,2 1,1 ` means the group id and canera id of image (./train/0_1_2.jpg) is 0 and 1. This image contains 5 members and there are 2 cloth-changing members within them. (Optionally, the cloth id of current cloth-changing members are also provided, as 1 and 1, respectively.)

## Download Link
* GroupPRCC dataset [Google Drive](https://drive.google.com/file/d/1m4O_G3Bdl9IBEYsLCJJnwLQLS8t3CtSR/view?usp=drive_link)
* GroupVC dataset [Google Drive](https://drive.google.com/file/d/1f0YFpND6iQkENabiD0-DR0LQQSp2HRxA/view?usp=drive_link)

## References
Our CCGReID datasets are based on serveral previous datasets.<br />      If you find our datasets useful for your research, please kindly cite the following papers.

```
@ARTICLE{PRCC_dataset,
  author={Yang, Qize and Wu, Ancong and Zheng, Wei-Shi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Person Re-Identification by Contour Sketch Under Moderate Clothing Change}, 
  year={2021},
  volume={43},
  number={6},
  pages={2029-2046},
}

@inproceedings{VC_dataset,
  title={When person re-identification meets changing clothes},
  author={Wan, Fangbin and Wu, Yang and Qian, Xuelin and Chen, Yixiong and Fu, Yanwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={830--831},
  year={2020}
}

@inproceedings{City1M_dataset,
  author={Zhang, Quan and Dang, Kaiheng and Lai, Jian-Huang and Feng, Zhanxiang and Xie, Xiaohua},
  title={Modeling 3D Layout for Group Re-Identification},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  month={June},
  year={2022},
  pages={7512-7520}
}
```

## License

* The datasets can only be used for ACADEMIC PURPOSES. NO COMERCIAL USE is allowed.
* Copyright © Sun Yat-sen University. All rights reserved.

