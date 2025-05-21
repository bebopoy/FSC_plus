
import argparse
import numpy as np
import torch
import torch.utils.data as Data
import sys, os
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.FSCSVD import Model
from datasets import ShapeNet
from utils.loss_utils import *

from models.model_utils import cross_attention, furthest_point_sample
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from config_pcn import cfg
import pdb

CATEGORIES_PCN       = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']

def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[1])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[1], size=n-pc.shape[1])])
    return pc[:, idx[:n], :]

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)

def test_single_category(category, model, params):
    test_dataset = ShapeNet(params.test_dataset_path, 'test_novel' if params.novel else 'test', category)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # Create directory for saving point cloud visualizations
    vis_dir = os.path.join(params.result_dir, params.exp_name, category)
    make_dir(vis_dir)

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    first_sample_saved = False  # Flag to save only the first sample
    with torch.no_grad():
        for p, c in test_dataloader:
            p = p.to(params.device)
            c = c.to(params.device)
            if params.test_point_num != 2048:
                partial = gather_points(c.transpose(1, 2).contiguous(), furthest_point_sample(c, params.test_point_num))
                p = random_sample(partial.transpose(1, 2).contiguous(), 2048)
            _, _, c_ = model(p)  # SVD decoder architecture
            cdl1, cdl2, f1 = calc_cd(c_, c, calc_f1=True)
            total_l1_cd += cdl1.mean().item()
            total_l2_cd += cdl2.mean().item()
            total_f_score += f1.mean().item()

            # Save point clouds for the first sample only
            if not first_sample_saved:
                for i in range(min(p.shape[0], 1)):  # Process only the first sample in the batch
                    input_pc = p[i].cpu().numpy()
                    gt_pc = c[i].cpu().numpy()
                    pred_pc = c_[i].cpu().numpy()

                    input_filename = os.path.join(vis_dir, f'sample_{category}_input.ply')
                    gt_filename = os.path.join(vis_dir, f'sample_{category}_gt.ply')
                    pred_filename = os.path.join(vis_dir, f'sample_{category}_pred.ply')

                    export_ply(input_filename, input_pc)
                    export_ply(gt_filename, gt_pc)
                    export_ply(pred_filename, pred_pc)

                first_sample_saved = True  # Ensure only the first sample is saved

    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    avg_f_score = total_f_score / len(test_dataset)

    return avg_l1_cd, avg_l2_cd, avg_f_score

def test(params):
    print(params.exp_name)

    # Load pretrained model
    model = Model(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(params.ckpt_path, weights_only=True)['model'])
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)', 'FScore-0.01(%)'))
    print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------'))

    if params.category == 'all':
        if params.novel:
            categories = CATEGORIES_PCN_NOVEL
        else:
            categories = CATEGORIES_PCN

        l1_cds, l2_cds, fscores = list(), list(), list()
        for category in categories:
            avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(category, model, params)
            print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))
            l1_cds.append(avg_l1_cd)
            l2_cds.append(avg_l2_cd)
            fscores.append(avg_f_score)

        print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------'))
        print('\033[32m{:20s}{:<20.4f}{:<20.4f}{:<20.4f}\033[0m'.format('Average', np.mean(l1_cds) * 1e3, np.mean(l2_cds) * 1e4, np.mean(fscores) * 1e2))
    else:
        avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(params.category, model, params)
        print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(params.category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=8, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--novel', type=bool, default=False, help='Unseen categories for testing')
    parser.add_argument('--test_point_num', type=int, default=2048, help='Number of point cloud when testing')
    params = parser.parse_args()

    if not params.emd:
        test(params)


# ### 修改说明
# 1. **限制保存第一个样本**：
#    - 添加了 `first_sample_saved` 标志，确保每个类别只保存第一个样本的点云数据。
#    - 在 `test_single_category` 中，使用 `min(p.shape[0], 1)` 确保只处理批次中的第一个样本（即使批次大小大于1）。
#    - 一旦保存了第一个样本的3个 PLY 文件（输入、真实、预测），`first_sample_saved` 设为 `True`，后续样本不再保存点云。

# 2. **文件名调整**：
#    - PLY 文件名改为 `sample_{category}_input.ply`、`sample_{category}_gt.ply` 和 `sample_{category}_pred.ply`，以类别名代替样本索引，避免跨类别混淆。
#    - 例如，对于类别 `airplane`，生成的文件为：
#      - `results/exp_name/airplane/sample_airplane_input.ply`
#      - `results/exp_name/airplane/sample_airplane_gt.ply`
#      - `results/exp_name/airplane/sample_airplane_pred.ply`

# 3. **输出文件总数**：
#    - 假设测试所有8个类别（`params.category == 'all'`），每个类别生成3个 PLY 文件（输入、真实、预测）。
#    - 总共生成：
#      \[
#      8 \times 3 = 24
#      \]
#      个 PLY 文件，分别保存在每个类别的子目录下（例如 `results/exp_name/airplane/`、 `results/exp_name/cabinet/` 等）。

# 4. **点云点数**：
#    - 每个 PLY 文件包含 **2048 个点**（由 `params.test_point_num` 和代码中的 `random_sample` 逻辑决定），每个点有 3 个坐标 (x, y, z)。

# ### 可视化查看方法
# - **保存位置**：文件保存在 `params.result_dir/params.exp_name/{category}/` 目录下，每个类别3个文件，共24个文件。
# - **查看工具**：
#   - 使用 **MeshLab** 或 **CloudCompare** 打开 PLY 文件，查看点云的3D结构。
#   - 使用 **Open3D** 脚本，例如：
#     ```python
#     import open3d as o3d
#     pcd_input = o3d.io.read_point_cloud("results/exp_name/airplane/sample_airplane_input.ply")
#     pcd_gt = o3d.io.read_point_cloud("results/exp_name/airplane/sample_airplane_gt.ply")
#     pcd_pred = o3d.io.read_point_cloud("results/exp_name/airplane/sample_airplane_pred.ply")
#     o3d.visualization.draw_geometries([pcd_input, pcd_gt, pcd_pred])
#     ```
#     你可以为不同点云设置颜色（例如输入为红色，真实为绿色，预测为蓝色）以便对比。

# ### 注意事项
# - **类别选择**：确保 `params.category` 设置为 `'all'` 以测试所有8个类别（`CATEGORIES_PCN` 或 `CATEGORIES_PCN_NOVEL`，取决于 `params.novel`）。如果只测试单一类别，则只生成3个 PLY 文件。
# - **数据集要求**：每个类别的测试数据集必须至少有1个样本，否则可能导致无文件生成。
# - **文件格式**：PLY 文件为 ASCII 格式，包含2048个点的坐标，兼容大多数点云可视化工具。
# - 如果需要交互式可视化（运行时直接显示点云）或其他格式（如添加颜色或合并点云），请告诉我，我可以进一步调整代码！

# 如果有其他问题或需要更具体的需求，请随时告知！