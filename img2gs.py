"""Single image -> 3DGS PLY + video. Thin wrapper around da3 CLI.

Usage:
    python img2gs.py photo.jpg
    python img2gs.py photo.jpg --outdir result --resolution 720
    python img2gs.py photo.jpg --model depth-anything/DA3-GIANT
"""

import argparse
import subprocess
import sys
from pathlib import Path

VENV_PYTHON = Path(__file__).parent / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    sys.exit(subprocess.call([str(VENV_PYTHON), __file__] + sys.argv[1:]))

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def main():
    p = argparse.ArgumentParser(description="Single image -> 3D Gaussian Splatting PLY + video")
    p.add_argument("image", help="Input image path")
    p.add_argument("-o", "--outdir", default="output", help="Output directory (default: output)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name or local path")
    p.add_argument("--resolution", type=int, default=504, help="Processing resolution (default: 504)")
    p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    args = p.parse_args()

    img = Path(args.image)
    if not img.is_file():
        p.error(f"{img} not found")

    cmd = [
        sys.executable, "-m", "depth_anything_3.cli", "image",
        str(img),
        "--model-dir", args.model,
        "--export-dir", args.outdir,
        "--export-format", "gs_ply-gs_video",
        "--infer-gs",
        "--device", args.device,
        "--process-res", str(args.resolution),
        "--auto-cleanup",
    ]

    print(f"Running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
