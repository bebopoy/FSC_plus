# FSC: Few-point Shape Completion

This repository contains the PyTorch implementation of the paper:

**[FSC: Few-point Shape Completion](https://arxiv.org/pdf/2403.07359), CVPR 2024**

<!-- <br> -->
Xianzu Wu, Xianfeng Wu, Tianyu Luan, Yajing Bai, Zhongyuan Lai, Junsong Yuan.
<!-- <br> -->
![example](Figure_2.pdf)

## Abstract

> While previous studies have demonstrated successful 3D object shape completion with a sufficient number of points, they often fail in scenarios when a few points, e.g. tens of points, are observed. Surprisingly, via entropy analysis, we find that even a few points, e.g. 64 points, could retain substantial information to help recover the 3D shape of the object. To address the challenge of shape completion with very sparse point clouds, we then propose Few-point Shape Completion (FSC) mode, which contains a novel dual-branch feature extractor for handling extremely sparse inputs, coupled with an extensive branch for maximal point utilization with a saliency branch for dynamic importance assignment. This model is further bolstered by a two-stage revision network that refines both the extracted features and the decoder output, enhancing the detail and authenticity of the completed point cloud. Our experiments demonstrate the feasibility of recovering 3D shapes from a few points. The proposed Few-point Shape Completion (FSC) mode model outperforms previous methods on both few-point inputs and many-point inputs, and shows good generalizability to different object categories.

## Pretrained Models
We provide pretrained FSC models(CVRR2024) on PCN and ShapeNet-55/34 [here](https://pan.baidu.com/s/1cOb827HaDLq0I-jHScakwg?pwd=iaw2) 
                                                                     code = iaw2.


## Get Started

### Requirement
- python >= 3.6
- PyTorch >= 1.8.0
- CUDA >= 11.1
- easydict
- opencv-python
- transform3d
- h5py
- timm
- open3d
- tensorboardX

Install PointNet++ and Density-aware Chamfer Distance.
```
cd pointnet2_ops_lib
python setup.py install

cd ../metrics/CD/chamfer3D/
python setup.py install

cd ../../EMD/
python setup.py install
```


### Dataset
Download the [PCN](https://gateway.infinitescript.com/s/ShapeNetCompletion) and [ShapeNet55/34](https://github.com/yuxumin/PoinTr) datasets, and specify the data path in config_*.py (pcn/55).
```
# PCN
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/path/to/ShapeNet/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/path/to/ShapeNet/%s/complete/%s/%s.pcd'

# ShapeNet-55
__C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH     = '/path/to/shapenet_pc/%s'

# Switch to ShapeNet-34 Seen/Unseen
__C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH       = '/path/to/datasets/ShapeNet34(ShapeNet-Unseen21)'
```
### Evaluation
```
# Specify the checkpoint path in config_*.py
__C.CONST.WEIGHTS = "path to your checkpoint"

python main_*.py --test (pcn/pcn_base)
```

### Training
```
python main_*.py (pcn/pcn_base) 
```

## Citation
```
@InProceedings{wu2024fsc,
  title={FSC: Few-point Shape Completion},
  author={Wu, Xianzu and Wu, Xianfeng and Luan, Tianyu and Bai, Yajing and Lai, Zhongyuan and Yuan, Junsong},
  journal={arXiv preprint arXiv:2403.07359},
  year={2024}
}
```


## Acknowledgement
The repository is based on [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), some of the code is borrowed from:
- [PytorchPointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [Density-aware Chamfer Distance](https://github.com/wutong16/Density_aware_Chamfer_Distance)
- [PoinTr](https://github.com/yuxumin/PoinTr)
- [GRNet](https://github.com/hzxie/GRNet)
- [PointAttN](https://github.com/ohhhyeahhh/PointAttN)
- [SVDFormer](https://github.com/czvvd/SVDFormer_PointSea)
- [SimpleView](https://github.com/princeton-vl/SimpleView)

The point clouds are visualized with [Easy3D](https://github.com/LiangliangNan/Easy3D).

We thank the authors for their great work！

<!-- ## License

This project is open sourced under MIT license. -->


