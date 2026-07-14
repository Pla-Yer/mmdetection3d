# BEVFusion + nuScenes-mini (Windows)

## 当前状态

- `data/v1.0-mini.tar` 已解压并整理到 `data/nuscenes/`
- 已生成：
  - `data/nuscenes/nuscenes_mini_infos_train.pkl`
  - `data/nuscenes/nuscenes_mini_infos_val.pkl`
  - `data/nuscenes/nuscenes_mini_dbinfos_train.pkl`
- 已补齐 Windows 兼容问题：
  - `nuScenes` sweep 路径分隔符
  - `numpy` 2.x 下 `np.long` 报错
  - `mmengine` 在 Windows 收集 `MSVC` 环境信息时报编码错误

## 可直接运行的命令

### 1. 数据准备

```bat
prepare_nuscenes_mini.cmd
```

如果你已经跑过一次并且 `pkl/dbinfos` 都在，可以跳过。

### 2. 1 iter 冒烟验证

```bat
smoke_bevfusion_nus_mini.cmd
```

这会启动 1 次训练迭代，并在 `work_dirs/bevfusion_nus_mini_smoke/` 下写出 `iter_1.pth`。

### 3. 训练 LiDAR-only BEVFusion

```bat
train_bevfusion_nus_mini_lidar.cmd
```

对应配置：

`projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py`

### 4. 训练 LiDAR-Cam 融合版

先准备两个权重：

- LiDAR-only 训练出来的 checkpoint
- `swint-nuimages-pretrained.pth`

由于你这台机器当前 Python/Windows 证书链会在在线下载时失败，建议手动下载 `swint-nuimages-pretrained.pth` 后放到本地，例如 `checkpoints\swint-nuimages-pretrained.pth`。

然后运行：

```bat
train_bevfusion_nus_mini_fusion.cmd work_dirs\bevfusion_nus_mini_lidar\epoch_6.pth checkpoints\swint-nuimages-pretrained.pth
```

对应配置：

`projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py`

### 5. 测试

测试融合版：

```bat
test_bevfusion_nus_mini.cmd work_dirs\bevfusion_nus_mini_fusion\epoch_6.pth fusion
```

测试 LiDAR-only：

```bat
test_bevfusion_nus_mini.cmd work_dirs\bevfusion_nus_mini_lidar\epoch_6.pth lidar
```

## 本次实际验证结果

- `mm3d` 环境可用，CUDA 可用，BEVFusion 自定义算子可正常导入
- `nuscenes_mini` 数据集可被 BEVFusion 的 LiDAR-only / LiDAR-Cam pipeline 真实读取
- `tools/train.py` 已真实启动
- `smoke` 训练已经产出：
  - `work_dirs/bevfusion_nus_mini_smoke/iter_1.pth`

## 说明

- `nuScenes-mini` 只适合做流程验证和小规模实验，不适合拿来对齐官方指标。
- `smoke_bevfusion_nus_mini.cmd` 默认会在 1 个训练 iter 后进入验证；如果你只想确认能起训，可以在运行时按 `Ctrl+C`，此时 `iter_1.pth` 已经写出。
