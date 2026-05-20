#!/usr/bin/env python3
"""双摄像头预览 —— 调视角 / 摆物体位置用，按 Q 退出。

用法:  python scripts/view_cameras.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

W, H, FPS = 640, 480, 30


def main():
    # -- 腕部 RealSense --
    serials = find_realsense_devices()
    wrist_serial = serials[0] if serials else ""
    wrist = RealSenseCamera(serial=wrist_serial, width=W, height=H, fps=FPS, enable_depth=False)

    # -- 全局 USB --
    global_cam = USBCamera(device_id="auto", width=W, height=H, fps=FPS)

    win = "Camera Preview | Wrist (L) + Global (R)  —  Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, W * 2, H)
    cv2.startWindowThread()

    print("Q / ESC = 退出")
    try:
        while True:
            wf = wrist.read()
            gf = global_cam.read()

            left = wf.rgb  # BGR
            right = cv2.resize(gf.rgb, (left.shape[1], left.shape[0]))

            # 十字准星
            for name, img in [("wrist", left), ("global", right)]:
                cx, cy = img.shape[1] // 2, img.shape[0] // 2
                cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
                cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
                cv2.putText(img, name, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            preview = np.hstack([left, right])
            cv2.imshow(win, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        wrist.close()
        global_cam.close()
        cv2.destroyAllWindows()
        print("退出。")


if __name__ == "__main__":
    main()
