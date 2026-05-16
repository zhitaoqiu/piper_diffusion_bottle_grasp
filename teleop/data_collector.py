#!/usr/bin/env python3
"""
Data collector for Piper bottle grasping (mirror mode).

Setup: leader + follower share one CAN bus (can0).
  - Human drags the leader arm by hand
  - Follower mirrors it automatically via CAN
  - We just read the follower state + cameras and record

Controls:
  SPACE    — start/stop recording an episode
  R        — discard current episode and restart recording
  E        — enable follower
  D        — disable follower
  ESC / Q  — quit

Usage:
  conda activate piper_act
  python3 teleop/data_collector.py
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from camera.rs_camera import (
    RealSenseCamera,
    USBCamera,
    describe_video_devices,
    find_realsense_devices,
    require_opencv,
)

cv2 = None

# --- Config ---
CAN_PORT = "can0"
CONTROL_RATE_HZ = 30
IMAGE_RATE_HZ = 15
VELOCITY_PCT = 50

DATASET_REPO = "piper/bottle_grasp"
DATASET_ROOT = str(PROJECT_ROOT / "data" / "lerobot_dataset")
TASK = "Grasp the bottle from the table"

WRIST_WIDTH, WRIST_HEIGHT, WRIST_FPS = 640, 480, 30
GLOBAL_WIDTH, GLOBAL_HEIGHT, GLOBAL_FPS = 640, 480, 30
GLOBAL_DEVICE_ID = "auto"  # SN0002 USB camera: scan /dev/video* by default
STATE_DIM = 7  # [j1..j6, gripper]


class ImageBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._wrist = None
        self._global = None

    def update(self, wrist=None, global_=None):
        with self._lock:
            if wrist is not None:
                self._wrist = wrist
            if global_ is not None:
                self._global = global_

    def get(self):
        with self._lock:
            return self._wrist, self._global


def camera_loop(wrist_cam, global_cam, buf: ImageBuffer, stop: threading.Event):
    period = 1.0 / IMAGE_RATE_HZ
    last_error_log = {"wrist": 0.0, "global": 0.0}

    def log_camera_error(name: str, exc: Exception):
        now = time.time()
        if now - last_error_log[name] > 2.0:
            print(f"  [WARN] {name} camera read failed: {exc}")
            last_error_log[name] = now

    while not stop.is_set():
        t0 = time.time()
        if wrist_cam is not None:
            try:
                buf.update(wrist=wrist_cam.read())
            except Exception as e:
                log_camera_error("wrist", e)
        if global_cam is not None:
            try:
                buf.update(global_=global_cam.read())
            except Exception as e:
                log_camera_error("global", e)
        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


def build_preview(wrist_frame, global_frame, enabled: bool, recording: bool, n_frames: int):
    preview = None
    if wrist_frame is not None:
        preview = cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR)
        h = preview.shape[0]
        if recording:
            cv2.circle(preview, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(preview, f"REC {n_frames}", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        status = "ENABLED" if enabled else "DISABLED"
        color = (0, 255, 0) if enabled else (0, 0, 255)
        cv2.putText(preview, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if global_frame is not None:
        g = cv2.cvtColor(global_frame.rgb, cv2.COLOR_RGB2BGR)
        if recording:
            cv2.circle(g, (30, 30), 12, (0, 0, 255), -1)
        if preview is not None:
            g = cv2.resize(g, (preview.shape[1], preview.shape[0]))
            preview = np.hstack([preview, g])
        else:
            preview = g

    return preview


def ensure_opencv():
    global cv2
    if cv2 is None:
        cv2 = require_opencv()
    return cv2


def check_gui_environment():
    """Print GUI diagnostics and fail early if DISPLAY is missing on Linux."""
    display = os.environ.get("DISPLAY", "")
    print(f"  DISPLAY={display if display else 'NOT SET'}")
    if sys.platform.startswith("linux") and not display:
        print("\n" + "=" * 60)
        print("  ERROR: DISPLAY is not set; OpenCV GUI windows cannot be shown.")
        print("  Are you running over SSH, inside Docker, or on a headless machine?")
        print("  Try: export DISPLAY=:0")
        print("=" * 60)
        raise RuntimeError("DISPLAY not set — cannot create OpenCV windows on Linux")
    print(f"  OpenCV available: {cv2 is not None}")


def safe_read_camera(name: str, cam, last_error_log: dict):
    """Read one frame from *cam*. Returns CameraFrame or None.

    Errors are printed with throttling (at most once every 2 s per camera).
    """
    if cam is None:
        return None
    try:
        return cam.read()
    except Exception as exc:
        now = time.time()
        if now - last_error_log.get(name, 0.0) > 2.0:
            print(f"  [WARN] {name} camera read failed: {exc}")
            last_error_log[name] = now
        return None


def create_preview_window(window_name: str, width: int = 1280, height: int = 480):
    """Create and show an initial OpenCV preview window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, width, height)
    test_img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(test_img, "Initializing...", (width // 3, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imshow(window_name, test_img)
    cv2.waitKey(1)
    # startWindowThread keeps the GUI alive even if main thread blocks
    cv2.startWindowThread()


def load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        print("[WARN] LeRobot not available - recording disabled")
        return None


def build_dataset_features():
    return {
        "observation.state": {
            "dtype": "float32", "shape": (STATE_DIM,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        },
        "action": {
            "dtype": "float32", "shape": (STATE_DIM,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video", "shape": (3, WRIST_HEIGHT, WRIST_WIDTH),
        },
        "observation.images.global_rgb": {
            "dtype": "video", "shape": (3, GLOBAL_HEIGHT, GLOBAL_WIDTH),
        },
    }


def has_episode_metadata(dataset_root: Path) -> bool:
    return any((dataset_root / "meta" / "episodes").glob("*/*.parquet"))


def move_incomplete_dataset(dataset_root: Path) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = dataset_root.with_name(f"{dataset_root.name}_incomplete_{stamp}")
    suffix = 1
    while backup.exists():
        backup = dataset_root.with_name(f"{dataset_root.name}_incomplete_{stamp}_{suffix}")
        suffix += 1
    dataset_root.rename(backup)
    return backup


def create_or_resume_dataset(LeRobotDataset, dataset_root: Path, repo_id: str = DATASET_REPO):
    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    features = build_dataset_features()

    if info_path.exists() and tasks_path.exists() and has_episode_metadata(dataset_root):
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=dataset_root)
        print(f"  Resumed existing dataset at {dataset_root}")
        return dataset

    if dataset_root.exists():
        backup = move_incomplete_dataset(dataset_root)
        print(f"  [WARN] Incomplete dataset moved to {backup}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=CONTROL_RATE_HZ,
        features=features, root=dataset_root, use_videos=True,
    )
    print(f"  Created new dataset at {dataset_root}")
    return dataset


def dataset_buffer_size(dataset) -> int:
    writer = getattr(dataset, "writer", None)
    if writer is None or writer.episode_buffer is None:
        return 0
    return int(writer.episode_buffer["size"])


def clear_dataset_buffer(dataset) -> None:
    if dataset is not None and dataset_buffer_size(dataset) > 0:
        dataset.clear_episode_buffer()


def should_quit(key: int, window_name: str | None = None) -> bool:
    if key in (27, ord('q'), ord('Q')):
        return True
    if window_name:
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return False
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-camera",
        default=os.environ.get("PIPER_GLOBAL_CAMERA", GLOBAL_DEVICE_ID),
        help="Global USB camera device: auto, /dev/videoX, or numeric index.",
    )
    parser.add_argument(
        "--wrist-serial",
        default=os.environ.get("PIPER_WRIST_SERIAL", ""),
        help="RealSense serial for the wrist camera. Empty means first detected.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List detected RealSense serials and /dev/video* nodes, then exit.",
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Open camera preview without connecting to Piper.",
    )
    parser.add_argument(
        "--no-wrist",
        action="store_true",
        help="Skip the wrist RealSense camera. Intended for --camera-only debugging.",
    )
    parser.add_argument(
        "--disable-motion-start-detect",
        action="store_true",
        help="Record immediately after SPACE instead of waiting for detected arm motion.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.005,
        help="Joint-space max delta threshold for motion-start detection.",
    )
    parser.add_argument(
        "--preroll-frames",
        type=int,
        default=5,
        help="Frames kept before detected motion when motion-start detection is enabled.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("PIPER_DATASET_ROOT", DATASET_ROOT),
        help="LeRobot dataset root for recording. Use a separate root for one-episode tests.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=os.environ.get("PIPER_DATASET_REPO", DATASET_REPO),
        help="LeRobot dataset repo_id stored in metadata.",
    )
    return parser.parse_args()


def print_camera_inventory():
    print(f"  RealSense: {find_realsense_devices()}")
    video_devices = describe_video_devices()
    if not video_devices:
        print("  Video devices: none")
        return
    print("  Video devices:")
    for device in video_devices:
        suffix = f"  ({device.name})" if device.name else ""
        print(f"    {device.path}{suffix}")


def init_cameras(args):
    print("\n[2/3] Initializing cameras ...")
    print_camera_inventory()
    rs_serials = find_realsense_devices()
    wrist_serial = args.wrist_serial or (rs_serials[0] if rs_serials else "")
    wrist_cam = None
    global_cam = None
    try:
        if args.no_wrist:
            print("  Wrist RealSense skipped.")
        else:
            wrist_cam = RealSenseCamera(
                serial=wrist_serial,
                width=WRIST_WIDTH, height=WRIST_HEIGHT, fps=WRIST_FPS, enable_depth=False,
            )
        global_cam = USBCamera(
            device_id=args.global_camera,
            width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT, fps=GLOBAL_FPS,
        )
        return wrist_cam, global_cam
    except Exception:
        if wrist_cam is not None:
            wrist_cam.close()
        if global_cam is not None:
            global_cam.close()
        raise


def run_camera_preview(wrist_cam, global_cam):
    window_name = "ACT Camera Preview | Wrist (L) + Global (R)"
    create_preview_window(window_name, 1280, 480)
    print(f"\n  Camera preview only. Q/ESC = quit")
    if wrist_cam is None and global_cam is None:
        print("  [WARN] Both cameras are None — no video source available.")
    print()

    error_log: dict[str, float] = {}
    last_missing_log = 0.0
    try:
        while True:
            wrist_frame = safe_read_camera("wrist", wrist_cam, error_log)
            global_frame = safe_read_camera("global", global_cam, error_log)

            preview = build_preview(wrist_frame, global_frame, True, False, 0)
            if preview is None:
                preview = np.zeros((480, 1280, 3), dtype=np.uint8)
                cv2.putText(preview, "Waiting for camera frames...", (400, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                # Periodically print why we have no preview
                now = time.time()
                if now - last_missing_log > 3.0:
                    reasons = []
                    if wrist_cam is None:
                        reasons.append("wrist camera not initialized")
                    elif wrist_frame is None:
                        reasons.append("wrist frame missing (check camera connection)")
                    if global_cam is None:
                        reasons.append("global camera not initialized")
                    elif global_frame is None:
                        reasons.append("global frame missing (check camera connection)")
                    print(f"  [INFO] No preview — " + "; ".join(reasons))
                    last_missing_log = now
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if should_quit(key, window_name):
                break
    finally:
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Piper Data Collector — Mirror Mode")
    print("=" * 60)

    if args.list_cameras:
        print_camera_inventory()
        return 0

    try:
        ensure_opencv()
    except ImportError as e:
        print(f"  FAIL: {e}")
        return 1

    try:
        check_gui_environment()
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return 1

    if args.camera_only:
        wrist_cam = None
        global_cam = None
        try:
            wrist_cam, global_cam = init_cameras(args)
            run_camera_preview(wrist_cam, global_cam)
        finally:
            if wrist_cam is not None:
                wrist_cam.close()
            if global_cam is not None:
                global_cam.close()
        return 0

    if args.no_wrist:
        print("  FAIL: --no-wrist is only supported together with --camera-only.")
        return 1

    # Create cv2 window BEFORE any hardware init, so we can isolate what breaks it
    window_name = "Data Collector | Wrist (L) + Global (R)"
    create_preview_window(window_name, 1280, 480)
    print(f"  Window '{window_name}' created.")

    # --- Robot ---
    from hardware.piper_wrapper import PiperRobot

    print("\n[1/3] Connecting Piper (can0) ...")
    robot = PiperRobot(can_port=CAN_PORT, gripper_exist=True)
    try:
        robot.connect()
        print("  Connected.")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # --- Cameras ---
    try:
        wrist_cam, global_cam = init_cameras(args)
    except Exception as e:
        print(f"  FAIL: {e}")
        robot.disconnect()
        return 1

    # Warm up cameras and verify frames are not black/dark
    print("  Warming up cameras ...")
    for i in range(15):
        try:
            wf = wrist_cam.read()
            gf = global_cam.read()
            w_mean = float(wf.rgb.mean())
            g_mean = float(gf.rgb.mean())
            if i == 0 or i == 14:
                print(f"  Frame {i+1}: wrist_mean={w_mean:.1f}, global_mean={g_mean:.1f}")
        except Exception as e:
            print(f"  [WARN] Camera warm-up frame {i+1} failed: {e}")
        time.sleep(0.05)
    print("  Cameras warmed up.")

    # --- LeRobot dataset ---
    print("\n[3/3] Setting up LeRobot dataset ...")
    LeRobotDataset = load_lerobot_dataset_class()
    if LeRobotDataset is not None:
        dataset_root = Path(args.dataset_root)
        try:
            dataset = create_or_resume_dataset(LeRobotDataset, dataset_root, args.dataset_repo_id)
        except Exception as e:
            print(f"  FAIL: {e}")
            robot.disconnect()
            wrist_cam.close()
            global_cam.close()
            return 1
    else:
        dataset = None

    print(
        f"  Timing: control={CONTROL_RATE_HZ}Hz, image_poll={IMAGE_RATE_HZ}Hz, "
        f"dataset_fps={getattr(dataset, 'fps', CONTROL_RATE_HZ) if dataset is not None else CONTROL_RATE_HZ}Hz"
    )
    if IMAGE_RATE_HZ != CONTROL_RATE_HZ:
        print("  [WARN] IMAGE_RATE_HZ differs from CONTROL_RATE_HZ; adjacent dataset frames may reuse images.")

    # --- State ---
    recording = False
    episode_count = getattr(dataset, "num_episodes", 0) if dataset is not None else 0
    prev_state = None  # for computing action = next state
    start_state = None
    motion_started = False
    motion_preroll = deque(maxlen=max(1, args.preroll_frames))
    wrist_frame = None
    global_frame = None
    camera_error_log: dict[str, float] = {}
    last_missing_log = 0.0

    print("\n" + "─" * 60)
    print("  SPACE = record/save    R = discard+restart")
    print("  E = enable             D = disable            Q/ESC = quit")
    print("  Return both arms to your fixed start pose manually before SPACE.")
    if not args.disable_motion_start_detect:
        print(
            f"  Motion-start detect: threshold={args.motion_threshold}, "
            f"pre-roll={max(1, args.preroll_frames)} frames"
        )
    print("─" * 60 + "\n")

    try:
        period = 1.0 / CONTROL_RATE_HZ
        img_interval = max(1, CONTROL_RATE_HZ // IMAGE_RATE_HZ)
        frame_idx = 0

        while True:
            t0 = time.time()

            # --- Read robot state ---
            cur_state = None
            try:
                cur_state = robot.get_joint_positions()
            except Exception as e:
                now = time.time()
                if now - camera_error_log.get("robot_state", 0.0) > 2.0:
                    print(f"  [WARN] robot.get_joint_positions failed: {e}")
                    camera_error_log["robot_state"] = now

            # --- Grab images (main thread, no threading) ---
            if frame_idx % img_interval == 0:
                wrist_frame = safe_read_camera("wrist", wrist_cam, camera_error_log)
                global_frame = safe_read_camera("global", global_cam, camera_error_log)

            # --- Record (action = next state) ---
            if recording and dataset is not None and cur_state is not None:
                if prev_state is not None and wrist_frame is not None and global_frame is not None:
                    frame = {
                        "observation.state": np.array(prev_state, dtype=np.float32),
                        "action": np.array(cur_state, dtype=np.float32),
                        "task": TASK,
                        "observation.images.wrist_rgb": np.transpose(wrist_frame.rgb, (2, 0, 1)),
                        "observation.images.global_rgb": np.transpose(global_frame.rgb, (2, 0, 1)),
                    }
                    if args.disable_motion_start_detect or motion_started:
                        try:
                            dataset.add_frame(frame)
                        except Exception as e:
                            print(f"  [WARN] add_frame: {e}")
                    else:
                        motion_preroll.append(frame)
                        if start_state is not None:
                            motion = float(
                                np.max(
                                    np.abs(
                                        np.asarray(cur_state[:6], dtype=np.float32)
                                        - np.asarray(start_state[:6], dtype=np.float32)
                                    )
                                )
                            )
                            if motion > args.motion_threshold:
                                motion_started = True
                                try:
                                    for buffered_frame in motion_preroll:
                                        dataset.add_frame(buffered_frame)
                                    print(
                                        f"  Motion detected at {motion:.4f}; "
                                        f"flushed {len(motion_preroll)} pre-roll frames."
                                    )
                                    motion_preroll.clear()
                                except Exception as e:
                                    print(f"  [WARN] add_frame: {e}")
                prev_state = cur_state
            elif recording and cur_state is None:
                now = time.time()
                if now - camera_error_log.get("record_skip", 0.0) > 2.0:
                    print("  [WARN] skip recording frame: robot state missing")
                    camera_error_log["record_skip"] = now

            # --- Preview ---
            preview = build_preview(wrist_frame, global_frame, robot.is_enabled, recording,
                                    dataset_buffer_size(dataset) if recording and dataset else 0)
            if preview is None:
                preview = np.zeros((480, 1280, 3), dtype=np.uint8)
                cv2.putText(preview, "Waiting for cameras...", (400, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                # Periodically explain WHY there is no preview
                now = time.time()
                if now - last_missing_log > 3.0:
                    reasons = []
                    if wrist_frame is None:
                        reasons.append("wrist frame missing")
                    if global_frame is None:
                        reasons.append("global frame missing")
                    if reasons:
                        print(f"  [INFO] No preview — " + "; ".join(reasons))
                    last_missing_log = now
            cv2.imshow(window_name, preview)

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if should_quit(key, window_name if preview is not None else None):
                break
            elif key == ord(' '):
                if not recording:
                    if not robot.is_enabled:
                        print("  [WARN] Press E to enable first!")
                    else:
                        recording = True
                        prev_state = None
                        start_state = cur_state
                        motion_started = args.disable_motion_start_detect
                        motion_preroll.clear()
                        if dataset:
                            clear_dataset_buffer(dataset)
                        print(f"\n  >>> Recording episode {episode_count + 1} ...")
                else:
                    recording = False
                    n_frames = dataset_buffer_size(dataset) if dataset is not None else 0
                    if dataset is not None and n_frames > 10:
                        dataset.save_episode()
                        episode_count += 1
                        print(f"  Saved episode {episode_count} ({n_frames} frames)")
                    else:
                        clear_dataset_buffer(dataset)
                        print("  Too short, discarded.")
                    start_state = None
                    motion_started = False
                    motion_preroll.clear()
            elif key in (ord('r'), ord('R')):
                if recording:
                    clear_dataset_buffer(dataset)
                    prev_state = None
                    start_state = cur_state
                    motion_started = args.disable_motion_start_detect
                    motion_preroll.clear()
                    print(f"  Discarded. Restarting episode {episode_count + 1} ...")
                else:
                    print("  [WARN] R only works while recording.")
            elif key == ord('e'):
                if not robot.is_enabled:
                    print("  Enabling ...")
                    print(f"  {'OK' if robot.enable(blocking=True) else 'FAILED'}")
            elif key == ord('d'):
                if robot.is_enabled:
                    robot.disable()
                    print("  Disabled.")

            frame_idx += 1
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        print("  Shutting down ...")
        if dataset is not None:
            print("  Finalizing dataset ...")
            try:
                dataset.finalize()
                print(f"  Done. Episodes: {episode_count}")
            except Exception as e:
                print(f"  [WARN] {e}")
        if robot.is_enabled:
            robot.disable()
        robot.disconnect()
        if wrist_cam is not None:
            wrist_cam.close()
        if global_cam is not None:
            global_cam.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
