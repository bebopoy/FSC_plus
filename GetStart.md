# 操作手册详情

    用于 SHU 数据挖掘课程学习。
                                                                                                                                ---by bebopoy 5/17

## Pretrained Models

FSC 原作者提供的最优模型文件：
We provide pretrained FSC models(CVRR2024) [PCN_Pretrained_Model](https://pan.baidu.com/s/1jzripjQKxOahAvymF9Vp7g?pwd=oq29) password: oq29

位置为：
FSC_best_PCN.pth （仓库根目录下）

---

## Get Started

### Requirement

一、

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

或去谷歌云下载现成的 conda python 包然后解压在对应目录下，便可直接使用。
[cuda_124 conda 环境包下载](https://drive.google.com/file/d/1bK--kCKqEJ9ke90QCFqdLGW0Oj7sSt9c/view?usp=sharing)

但要求比较严苛：

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.07                 Driver Version: 566.07         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   54C    P8             14W /  115W |    1709MiB /   6144MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

```

二、

Install PointNet++ and Density-aware Chamfer Distance.（这一步需要编译环境无误， VS 软件 c++拓展与 CUDA 适配）
对应的配置文件经过覆写，可以适应 NVD 30 系列显卡

```
cd pointnet2_ops_lib
python setup.py install

cd ../metrics/CD/chamfer3D/
python setup.py install

cd ../../EMD/
python setup.py install
```

---

### Dataset

训练数据：
Download the [PCN](https://gateway.infinitescript.com/s/ShapeNetCompletion)，并且修改 config_pcn.py 内容

位置为：
datasets\ShapeNetCompletion

测试数据：
Download the [PCN](https://drive.google.com/file/d/1OvvRyx02-C_DkzYiJ5stpin0mnXydHQ7/view?usp=sharing)

位置为：
datasets\PCN

---

### Command for Train

```
python main_pcn.py
```

---

### Command for Test

--test True
--exp_name BEST 记录各个类别第一个输入、输出结果、以及真实点云信息，用于可视化
--ckpt_path FSC_best_PCN.pth 用于选择模型
--test_dataset_path E:/projest_in_les/FSC/datasets/PCN 选测试集
--category all 选择测试的补全类行
--novel False

```
（models\FSCSVD.py中的内容，默认可以使用原论文FSC模型）
测原论文的最优模型：(FSC)
python main_pcn.py --test True --category all --test_dataset_path E:/projest_in_les/FSC/datasets/PCN --novel False --ckpt_path FSC_best_PCN.pth --exp_name BEST

（models\FSCSVD.py中的内容，默认可以使用原论文FSC模型）
测训练的模型：（FSC）
python main_pcn.py --test True --category all --test_dataset_path E:/projest_in_les/FSC/datasets/PCN --novel False --ckpt_path FSC_PCN\checkpoints\2025-05-07_00-06-21\ckpt-best.pth(your FSC modelpath) --exp_name RAW

（使用models\new_FSCSVD.py 去替换models\FSCSVD.py中的内容，才可以使用FSC++模型）
测训练的模型：（FSC++）
python main_pcn.py --test True --category all --test_dataset_path E:/projest_in_les/FSC/datasets/PCN --novel False --ckpt_path FSC_PCN\checkpoints\2025-05-07_00-06-21\ckpt-best.pth(your FSC++ modelpath) --exp_name IMP

```

---

### 相关链接

- 仓库地址：https://github.com/bebopoy/FSC_plus.git
- 仓库配置手册：https://github.com/bebopoy/FSC_plus/blob/main/GetStart.md
- Raw FSC 训练日志（含可视化）：https://wandb.ai/xiangtongnie-shu/FSC/runs/y4yvxkwf
- Imp FSC++训练日志（含可视化）：https://wandb.ai/xiangtongnie-shu/FSC/runs/uzcqpe3n
- FSC 与 FSC++测试日志：https://drive.google.com/drive/folders/1yQ6TTl-7kdmh-Roj6_pF2GkB8M0fBR7u?usp=sharing
