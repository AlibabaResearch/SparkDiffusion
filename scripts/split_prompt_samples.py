"""
Split horizontally-concatenated per-prompt videos into individual sample files.

Input  : 0_ema_Sample_Iter{iter}_prompt{idx:02d}.mp4
         (N samples laid side-by-side along the width axis)
Output : 0_ema_Sample_Iter{iter}_prompt{idx:02d}_s{s:02d}.mp4
         (one file per sample, already-split files are skipped)

Usage:
    python scripts/split_prompt_samples.py \
        --dir outputs/sparkdiffusion/Wan/.../EveryNDrawSample_Distill \
        --num_samples 5

The script auto-detects per-sample width as W // num_samples and uses ffmpeg
crop filter for lossless-ish splitting (re-encodes with libx264 crf 0 to avoid
re-encoding artefacts; or --copy to stream-copy without re-encoding).
"""

import argparse
import glob
import os
import re
import subprocess
import sys


def get_video_wh(path: str):
    """Return (width, height) of a video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def split_video(src: str, out_prefix: str, num_samples: int, stream_copy: bool, dry_run: bool):
    """Crop ``src`` into ``num_samples`` equal-width slices."""
    w, h = get_video_wh(src)
    if w % num_samples != 0:
        print(f"  [WARN] W={w} not divisible by {num_samples}, skipping {src}")
        return
    per_w = w // num_samples

    for s in range(num_samples):
        out_path = f"{out_prefix}_s{s:02d}.mp4"
        if os.path.exists(out_path):
            print(f"  [SKIP] {os.path.basename(out_path)} already exists")
            continue
        x_offset = s * per_w
        vf = f"crop={per_w}:{h}:{x_offset}:0"
        if stream_copy:
            cmd = ["ffmpeg", "-y", "-i", src, "-vf", vf, "-c:v", "libx264",
                   "-preset", "fast", "-crf", "0", "-an", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", src, "-vf", vf,
                   "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", out_path]
        if dry_run:
            print(f"  [DRY] {' '.join(cmd)}")
        else:
            print(f"  -> {os.path.basename(out_path)}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="Split per-prompt sample videos")
    parser.add_argument("--dir", required=True, help="Path to EveryNDrawSample_Distill directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per video (default: 5)")
    parser.add_argument("--copy", action="store_true", help="Use lossless crf=0 re-encode (default: crf=18)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    # Pattern: 0_ema_Sample_Iter000001000_prompt00.mp4
    # Exclude already-split files (_sNN suffix).
    pattern = os.path.join(args.dir, "0_ema_Sample_Iter*_prompt*.mp4")
    files = sorted(glob.glob(pattern))
    # Filter out _sNN files
    files = [f for f in files if not re.search(r"_s\d{2}\.mp4$", f)]

    if not files:
        print(f"No prompt mp4 files found in: {args.dir}")
        sys.exit(0)

    print(f"Found {len(files)} prompt video(s) in {args.dir}")
    for f in files:
        base = f[:-4]  # strip .mp4
        print(f"Splitting: {os.path.basename(f)}")
        split_video(f, base, args.num_samples, args.copy, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
