# FullMatch

**Boosting Semi-supervised Learning by Exploiting All Unlabeled Data**

Yuhao Chen, Xin Tan, Borui Zhao, Zhaowei Chen, Renjie Song, Jiajun Liang, Xuequan Lu

CVPR 2023, [Arxiv](https://arxiv.org/abs/2303.11066)

This repo is the Megengine implementation of FullMatch. The Pytorch implementation will be released soon.

### Experiment

1. Install MegEngine (version==1.12.2/1.12.3)

2. For training FullMatch:
```bash
python fullmatch.py --c config/fullmatch/fullmatch_cifar100.yaml
```
3. For training FullFlex:
```bash
python fullflex.py --c config/fullflex/fullflex_cifar100.yaml
```


### Note

Since the official Megengine does not support many classification benchmarks (e.g., SVHN, STL10), we will release them in the Pytorch implementation.

We thanks the [TorchSSL](https://github.com/TorchSSL/TorchSSL) project for reference.

### Log & Models

We'll upload them within a week.

### Liscense
FullMatch is released under the Apache 2.0 license. See [LICENSE](license) for details.

### Citation
```
@inproceedings{chen2023boosting,
  title={Boosting Semi-Supervised Learning by Exploiting All Unlabeled Data},
  author={Chen, Yuhao and Tan, Xin and Zhao, Borui and Chen, Zhaowei and Song, Renjie and Liang, Jiajun and Lu, Xuequan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7548--7557},
  year={2023}
}
```