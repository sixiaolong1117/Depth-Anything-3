"""Real-time webcam depth estimation using Depth Anything 3.

Usage:
    python cam_depth.py
    python cam_depth.py --model depth-anything/DA3-LARGE-1.1 --camera 0
    python cam_depth.py --resolution 384 --colormap inferno
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path

VENV_PYTHON = Path(__file__).parent / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    sys.exit(subprocess.call([str(VENV_PYTHON), __file__] + sys.argv[1:]))

import cv2
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def main():
    p = argparse.ArgumentParser(description="Real-time webcam depth estimation")
    p.add_argument("--model", default="depth-anything/DA3-LARGE-1.1",
                   help="Model name (default: DA3-LARGE-1.1, faster than GIANT)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--resolution", type=int, default=384, help="Processing resolution (default: 384)")
    p.add_argument("--colormap", default="Spectral", help="Matplotlib colormap (default: Spectral)")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    p.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    args = p.parse_args()

    print(f"Loading model: {args.model} ...")
    model = DepthAnything3.from_pretrained(args.model).to(args.device).eval()
    print("Model loaded.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        p.error(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Press 'q' to quit, 's' to save screenshot.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()

        with torch.no_grad():
            prediction = model.inference(
                [Image.fromarray(frame_rgb)],
                process_res=args.resolution,
            )

        depth = prediction.depth[0]  # (H, W)
        depth_vis = visualize_depth(depth, cmap=args.colormap)
        depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(depth_bgr, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # side-by-side: original | depth
        h, w = frame_bgr.shape[:2]
        depth_bgr = cv2.resize(depth_bgr, (w, h))
        combined = np.hstack([frame_bgr, depth_bgr])
        cv2.imshow("Original | Depth (press q to quit, s to save)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("depth_screenshot.png", combined)
            print("Saved depth_screenshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
