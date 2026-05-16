# Piper Diffusion Policy Bottle Grasp

基于 Piper 双臂 + Diffusion Policy 算法的瓶子抓取项目。

## 硬件配置

| 组件 | 型号 | 连接方式 | 用途 |
|---|---|---|---|
| 被控臂 | Piper | CAN (can0) | 执行抓取动作 |
| 示教臂 | Piper | 同一条 CAN 总线 | 人手拖动，被控臂自动镜像跟随 |
| 腕部相机 | RealSense D435i | USB 3.0 | 末端 RGB |
| 全局相机 | USB SN0002 | USB | 俯视 RGB |
| GPU | NVIDIA RTX 3060 12GB | — | 训练 + 推理 |

> **镜像模式**：示教臂和被控臂共享同一条 CAN 总线（can0），由硬件层面自动完成镜像跟随，无需软件转发。采集时只需读取被控臂状态即可。

## 环境搭建

```bash
conda create -n piper_act python=3.10 -y
conda activate piper_act
pip install -r requirements.txt
```

关键依赖：
- `lerobot` — LeRobot 框架（含 Diffusion Policy）
- `torch >= 2.0`（CUDA 版）
- `pyrealsense2`（RealSense 相机）
- `opencv-python`（图像处理）
- `numpy < 2.0`（避免与 cv2 的 NumPy ABI 冲突）

## 项目结构

```
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml              # 参考配置
│   └── start_pose.json           # 起始位姿
├── hardware/
│   └── piper_wrapper.py          # Piper 机械臂 SDK 封装
├── camera/
│   └── rs_camera.py              # RealSense + USB 相机驱动
├── teleop/
│   └── data_collector.py         # 示教数据采集（LeRobot 格式）
├── training/
│   └── train.sh                  # Diffusion Policy 训练脚本
├── inference/
│   ├── deploy.py                 # 真机推理部署
│   └── eval.py                   # 离线评估（MSE）
├── scripts/
│   ├── analyze_dataset_motion.py # 数据集运动分析
│   ├── rebuild_trimmed_dataset.py# 数据集裁剪重建
│   ├── setup_can.sh              # CAN 总线配置
│   └── setup_env.sh              # 环境安装
├── test_hardware.py              # 硬件链路验证
└── data/                         # LeRobot 数据集
```

## 完整工作流

### Step 1 — 验证硬件

```bash
conda activate piper_act
python3 test_hardware.py
python3 teleop/data_collector.py --list-cameras
python3 teleop/data_collector.py --camera-only --global-camera auto
```

### Step 2 — 采集示教数据

```bash
conda activate piper_act
python3 teleop/data_collector.py --global-camera auto
```

| 按键 | 功能 |
|---|---|
| E | 使能被控臂 |
| 空格 | 开始/停止录制 |
| R | 丢弃当前条重录 |
| Q | 退出 |

> 每条数据开始前，手动把示教臂和被控臂一起回到固定起点，再按空格录制。建议采集 **50-100 条**，变化瓶子位置和角度。

### Step 3 — 训练 Diffusion Policy

```bash
conda activate piper_act
nohup bash training/train.sh > /tmp/train_piper_diffusion.log 2>&1 &
tail -f /tmp/train_piper_diffusion.log
```

checkpoint 保存在 `outputs/train/piper_bottle_grasp/checkpoints/`。

### Step 4 — 离线评估

```bash
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

### Step 5 — 真机部署

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

按**空格**执行一次抓取。可选 `--velocity-pct 30` 调低速度。

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/050000/pretrained_model \
    --debug-actions \
    --debug-every 1
```

## 已知问题

### NumPy 版本

`numpy<2.0` 是必须的。OpenCV 的 Python 包编译时链接了 NumPy 1.x ABI。如果出现 `_ARRAY_API` 错误：

```bash
pip install "numpy<2" --force-reinstall opencv-python
```

### PYTHONPATH 污染

如果系统有 ROS2，激活 conda 环境后系统 Python 包可能被加载。需在 conda hooks 中处理。

### USB 摄像头无法打开

用 `--list-cameras` 列出设备，再用 `--global-camera N` 指定设备号。

## SDK API 速查

`PiperRobot`（`hardware/piper_wrapper.py`）：

| 方法 | 说明 |
|---|---|
| `connect()` | 连接 CAN 并使能 |
| `enable()` / `disable()` | 使能/失能电机 |
| `get_joint_positions()` | 读取 [j1..j6, gripper]，单位 rad / m |
| `set_joint_positions(pos, velocity_pct)` | 下发关节指令 |
| `disconnect()` | 断开 CAN |
