"""RealSense D435/D435i + USB camera wrappers for synchronized capture."""

import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

cv2 = None
HAS_OPENCV = False
OPENCV_IMPORT_ERROR = None


def require_opencv():
    """Import OpenCV on demand so camera listing works even if cv2 is broken."""
    global cv2, HAS_OPENCV, OPENCV_IMPORT_ERROR
    if HAS_OPENCV:
        return cv2
    try:
        import cv2 as cv2_module
    except Exception as exc:
        OPENCV_IMPORT_ERROR = exc
        raise ImportError(
            "opencv-python failed to import. If you see a NumPy ABI error, "
            "install a NumPy 1.x version, for example: pip install 'numpy<2'."
        ) from exc
    cv2 = cv2_module
    HAS_OPENCV = True
    return cv2


@dataclass
class CameraFrame:
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: Optional[np.ndarray]  # (H, W) float32 in metres, or None
    timestamp: float


@dataclass(frozen=True)
class VideoDeviceInfo:
    path: str
    index: Optional[int]
    name: str


class RealSenseCamera:
    """Intel RealSense D435/D435i depth camera."""

    def __init__(
        self,
        serial: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
    ):
        if not HAS_REALSENSE:
            raise ImportError("pyrealsense2 not installed")

        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = rs.pipeline()
        self._align = None

        # Start with RGB first (always works), then try adding depth
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        except Exception:
            print("  Depth stream not available, running RGB-only")
            enable_depth = False

        self._profile = self._pipeline.start(config)

        if enable_depth:
            try:
                self._align = rs.align(rs.stream.color)
                # Quick warm-up
                for _ in range(10):
                    self._pipeline.wait_for_frames(5000)
            except RuntimeError:
                self._pipeline.stop()
                config = rs.config()
                if serial:
                    config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self._profile = self._pipeline.start(config)
                self._align = None
                enable_depth = False
                print("  Depth failed, falling back to RGB-only")
        else:
            for _ in range(10):
                self._pipeline.wait_for_frames(5000)

        self.enable_depth = enable_depth
        print(f"  RealSense ready ({width}x{height} @ {fps}fps, depth={enable_depth})")

    def read(self) -> CameraFrame:
        frames = self._pipeline.wait_for_frames()
        ts = time.time()

        if self._align:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())  # BGR from camera

        depth = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001

        return CameraFrame(rgb=rgb, depth=depth, timestamp=ts)

    def close(self) -> None:
        self._pipeline.stop()


def _video_sort_key(path: Path) -> tuple:
    match = re.search(r"(\d+)$", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def _video_index(value: Union[int, str]) -> Optional[int]:
    if isinstance(value, int):
        return value
    match = re.search(r"(?:^|/)video(\d+)$", value.strip())
    if match:
        return int(match.group(1))
    if value.strip().isdigit():
        return int(value.strip())
    return None


def describe_video_devices() -> list[VideoDeviceInfo]:
    """List Linux video nodes with their sysfs display names when available."""
    devices: list[VideoDeviceInfo] = []
    for path in sorted(Path("/dev").glob("video*"), key=_video_sort_key):
        name_path = Path("/sys/class/video4linux") / path.name / "name"
        try:
            name = name_path.read_text(encoding="utf-8").strip()
        except OSError:
            name = ""
        devices.append(VideoDeviceInfo(path=str(path), index=_video_index(str(path)), name=name))
    return devices


def find_usb_video_devices() -> list[str]:
    """List Linux video device nodes that OpenCV can try to open."""
    return [device.path for device in describe_video_devices()]


def _looks_like_realsense(device: VideoDeviceInfo) -> bool:
    name = device.name.lower()
    return "realsense" in name or "intel" in name


def _auto_video_candidates() -> list[str]:
    devices = describe_video_devices()
    preferred = [device for device in devices if not _looks_like_realsense(device)]
    fallback = [device for device in devices if _looks_like_realsense(device)]
    ordered = preferred + fallback if preferred else devices
    return [device.path for device in ordered]


def _normalize_device_id(device_id: Union[int, str, None]) -> Union[int, str, None]:
    if isinstance(device_id, str):
        value = device_id.strip()
        if value.lower() in {"", "auto", "none"}:
            return None
        if value.isdigit():
            return int(value)
        return value
    return device_id


class USBCamera:
    """Generic USB camera via OpenCV VideoCapture."""

    def __init__(
        self,
        device_id: Union[int, str, None] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
    ):
        cv2_module = require_opencv()

        requested_id = _normalize_device_id(device_id)
        candidates = _auto_video_candidates() if requested_id is None else [requested_id]
        if not candidates:
            raise IOError(
                "No USB camera candidates found. Check /dev/video* or pass "
                "--global-camera 6 or --global-camera /dev/video6."
            )
        if requested_id is None:
            print(f"  USB auto candidates: {candidates}")

        errors: list[str] = []
        self.cap = None
        self.device_id = None

        for candidate in candidates:
            cap = self._open(candidate)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{candidate}: open failed")
                continue

            if fourcc:
                cap.set(cv2_module.CAP_PROP_FOURCC, cv2_module.VideoWriter_fourcc(*fourcc))
            cap.set(cv2_module.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2_module.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2_module.CAP_PROP_FPS, fps)

            frame = None
            for _ in range(20):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size:
                    break
                time.sleep(0.03)
            else:
                cap.release()
                errors.append(f"{candidate}: opened but read returned no frames")
                continue

            self.cap = cap
            self.device_id = candidate
            actual_w = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2_module.CAP_PROP_FPS)
            backend = self._backend_name(cap)
            print(
                f"  USB camera {candidate} ready "
                f"({actual_w}x{actual_h} @ {actual_fps:.1f}fps, backend={backend})"
            )
            return

        detail = "; ".join(errors) if errors else "no candidates were tested"
        raise IOError(f"Unable to open USB camera ({detail})")

    @staticmethod
    def _open(device_id: Union[int, str]):
        cv2_module = require_opencv()
        open_id = _video_index(device_id) if isinstance(device_id, str) else device_id
        if open_id is None:
            open_id = device_id
        if sys.platform.startswith("linux") and hasattr(cv2_module, "CAP_V4L2"):
            cap = cv2_module.VideoCapture(open_id, cv2_module.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2_module.VideoCapture(open_id)

    @staticmethod
    def _backend_name(cap) -> str:
        try:
            return cap.getBackendName()
        except Exception:
            return "unknown"

    def read(self) -> CameraFrame:
        ret, frame = self.cap.read()
        if not ret:
            raise IOError(f"USB camera {self.device_id} read failed")
        return CameraFrame(rgb=frame, depth=None, timestamp=time.time())

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def find_realsense_devices() -> list:
    """List connected RealSense devices. Returns list of serial numbers."""
    if not HAS_REALSENSE:
        return []
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        return [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]
    except Exception as exc:
        print(f"  [WARN] RealSense device query failed: {exc}")
        return []
