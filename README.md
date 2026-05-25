# Piper Diffusion Policy — Bottle Pick & Place Aside

基于 Piper 双臂 + Diffusion Policy（LeRobot 平台）的瓶子抓取并放到一边项目。

## 硬件配置

| 组件 | 型号 | 连接方式 | 用途 |
|---|---|---|---|
| 主臂（示教） | Piper | CAN (can0) | 人手拖动，从臂自动镜像跟随 |
| 从臂（执行） | Piper | CAN (can0) | 数据采集时镜像主臂；推理时由 Diffusion Policy 控制 |
| 腕部相机 | RealSense D435i | USB 3.0 | 末端 RGB |
| 全局相机 | USB SN0002 | USB | 俯视 RGB |
| GPU | NVIDIA RTX 3060 12GB | — | 训练 + 推理 |

> **镜像模式**：主臂和从臂共享同一条 CAN 总线（can0），由硬件层面自动完成镜像跟随，无需软件转发。采集时只需读取从臂状态即可。
> **推理阶段**：只需连接从臂，Diffusion Policy 直接输出从臂关节目标。

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
│   ├── default.yaml              # 参考配置（双臂采集 + 单臂推理）
│   └── start_pose.json           # 起始位姿
├── hardware/
│   ├── config_piper.py           # Piper 配置 dataclass
│   └── piper_wrapper.py          # Piper 机械臂 LeRobot SDK 封装
├── camera/
│   └── rs_camera.py              # RealSense + USB 相机驱动
├── teleop/
│   └── data_collector.py         # 示教数据采集（LeRobot 格式）
├── training/
│   └── train.sh                  # Diffusion Policy 训练脚本
├── inference/
│   ├── deploy.py                 # 真机推理部署（Diffusion Policy）
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
| E | 使能从臂 |
| 空格 | 开始/停止录制 |
| R | 丢弃当前条重录 |
| D | 失能从臂 |
| Q/ESC | 退出 |

> 每条数据开始前，手动把主臂和从臂一起回到固定起点，再按空格录制。动作流程：靠近瓶子 → 夹爪闭合抓取 → 抬起 → 平移放置 → 打开夹爪 → 离开。建议采集 **50-100 条**，变化瓶子位置和角度。

### Step 3 — 训练 Diffusion Policy

```bash
conda activate piper_act
nohup bash training/train.sh > /tmp/train_piper_diffusion.log 2>&1 &
tail -f /tmp/train_piper_diffusion.log
```

checkpoint 保存在 `outputs/train/piper_bottle_pick_place_aside/checkpoints/`。
训练使用 LeRobot 内置的 Diffusion Policy（DiT backbone），关键参数：horizon=8, n_action_steps=4, n_obs_steps=2。

### Step 4 — 离线评估

```bash
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_pick_place_aside/checkpoints/last/pretrained_model
```

### Step 5 — 真机部署

```bash
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_pick_place_aside/checkpoints/last/pretrained_model
```

按**空格**执行一次推理。部署脚本默认要求 CUDA 可用；如果 `nvidia-smi`
无法正常显示显卡/驱动，脚本会直接退出，避免自动落到 CPU 把电脑卡死。

建议先做一次轻量检查：

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

可选参数：

```bash
# 调低速度
python3 inference/deploy.py --checkpt <path> --velocity-pct 25

# 加快调试推理；可能略降策略质量，适合 smoke test
python3 inference/deploy.py --checkpt <path> --num-inference-steps 16

# 调试模式：打印每步动作
python3 inference/deploy.py --checkpt <path> --debug-actions --debug-every 5

# Dry run：预览但不发送指令
python3 inference/deploy.py --checkpt <path> --dry-run

# CPU 只建议 dry-run/排错；默认禁用，防止整机无响应
python3 inference/deploy.py --checkpt <path> --allow-cpu --device cpu --dry-run --num-inference-steps 4

# 无 GUI 模式（SSH/Docker）
python3 inference/deploy.py --checkpt <path> --no-gui

# 保存 rollout 数据
python3 inference/deploy.py --checkpt <path> --save-rollout
```

## ACT adapter_v2 数据路线

ACT 项目里已经验证过一条更稳定的 `adapter_v2` 路线：

- 固定起点，7 维绝对关节目标 `[j1..j6, gripper]`
- 单全局相机 `observation.images.global_rgb`
- 10 条基线数据 + 15 条补采数据，默认剔除补采数据中的坏 `episode 9`

当前仓库用重建导入方式生成 Diffusion 训练集，导入后 LeRobot 会重新
计算 24 条数据的 `stats.json`：

```bash
conda activate piper_act
python3 scripts/import_adapter_v2_24demo.py
```

导入默认读取：

```text
/home/huatec/piper_act_bottle_grasp/data/lerobot_dataset_piper_bottle_adapter_v2_10demo
/home/huatec/piper_act_bottle_grasp/data/lerobot_dataset_piper_bottle_adapter_v2_new_demos
```

输出到：

```text
data/lerobot_dataset_piper_bottle_adapter_v2_24demo
```

训练 Diffusion：

```bash
bash training/train_adapter_v2_diffusion.sh
```

先做离线评估：

```bash
python3 inference/eval.py \
    --checkpt outputs/train/diffusion_adapter_v2_24demo/checkpoints/last/pretrained_model \
    --dataset-root data/lerobot_dataset_piper_bottle_adapter_v2_24demo \
    --dataset-repo-id piper/adapter_v2_24demo_diffusion
```

`inference/deploy.py` 仍保留当前 approach-only 安全逻辑，会强制保持夹爪
打开。训练 `adapter_v2` 完整轨迹 checkpoint 后，使用单独入口让 Diffusion
控制夹爪和完整轨迹：

每次真机测试前，先回到 `adapter_v2` 起点并打开夹爪：

```bash
python3 scripts/go_adapter_v2_start.py \
    --can-port can0 \
    --velocity-pct 20 \
    -y
```

这个脚本读取 `config/adapter_v2_start_pose.json`，不要用旧的
`scripts/go_home.py`；旧脚本对应的是 `config/start_pose.json`，夹爪接近闭合。

```bash
python3 inference/deploy_adapter_v2.py \
    --checkpt outputs/train/diffusion_adapter_v2_24demo_5k/checkpoints/002000/pretrained_model \
    --global-camera /dev/video6 \
    --can-port can0 \
    --velocity-pct 10 \
    --max-steps 120 \
    --num-inference-steps 16 \
    --debug-actions
```

这个入口默认会用 `config/adapter_v2_start_pose.json` 做起点守卫，未回到
`adapter_v2` 起始区域时不会发起一条新轨迹。建议先加 `--dry-run` 看动作
输出，再做真机执行。部署入口还默认启用夹爪开口门控：起步后的前 35 个
policy step，且机械臂离开起点不足 0.08 rad 前，都会保持夹爪打开，避免
完整轨迹模型一启动就合爪。

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
