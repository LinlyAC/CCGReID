# CCGReID-datasets
Cloth-changing group re-identification dataset, incluidng GroupPRCC and GroupVC datasets.



## Introduction

* GroupPRCC dataset contains 7,514 training images of 64 group identities and 3,676 testing images of 32 group identities, which are captured under 3 cameras.
* GroupVC dataset contains 3,832 training images of 111 group identities and 4,179 testing images of 118 group identities, which are captured under 4 cameras.
* Both datasets provide SCS (same-cloth setting) and CCS (cross-cloth setting) protocols.

Please note: For ease of storage and distribution, we store the member (size: $256(height)\times128(width)$) in each group in a single, uniform-sized image (size: $ 256(height)\times256(width) $).

## Annotation

This sentence uses `$` delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$
This sentence uses $\` and \`$ delimiters to show math inline:  $`\sqrt{3x-1}+(1+x)^2`$

## Download Link
* GroupPRCC dataset [Google Drive](https://drive.google.com/file/d/1m4O_G3Bdl9IBEYsLCJJnwLQLS8t3CtSR/view?usp=drive_link)
* GroupVC dataset [Google Drive](https://drive.google.com/file/d/1f0YFpND6iQkENabiD0-DR0LQQSp2HRxA/view?usp=drive_link)

## References
Our CCGReID datasets are based on serveral previous datasets. If you find our datasets useful for your research, please kindly cite the following papers.
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

