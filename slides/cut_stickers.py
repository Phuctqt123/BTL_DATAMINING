"""Cắt bảng sticker 3x2 thành từng hình, làm trong suốt phần nền trắng nối với viền ảnh.

python slides/cut_stickers.py slides/gen/sheet1.png slides/gen/sticker
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def transparent_bg(im: Image.Image, thr: int = 238) -> Image.Image:
    a = np.asarray(im.convert("RGB")).astype(int)
    h, w, _ = a.shape
    white = (a.min(axis=2) >= thr)
    bg = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if white[y, x] and not bg[y, x]:
                bg[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if white[y, x] and not bg[y, x]:
                bg[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and white[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True; q.append((ny, nx))
    rgba = np.dstack([a.astype(np.uint8), np.where(bg, 0, 255).astype(np.uint8)])
    out = Image.fromarray(rgba, "RGBA")
    return out.crop(out.getbbox())


def main(src, prefix, cols=3, rows=2):
    im = Image.open(src).convert("RGB")
    W, H = im.size
    for r in range(rows):
        for c in range(cols):
            cell = im.crop((c * W // cols, r * H // rows, (c + 1) * W // cols, (r + 1) * H // rows))
            out = transparent_bg(cell)
            p = Path(f"{prefix}_{r * cols + c + 1}.png")
            out.save(p)
            print(p, out.size)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) == 3 and sys.argv[2].endswith(".png"):
        out = transparent_bg(Image.open(sys.argv[1]))
        out.save(sys.argv[2]); print(sys.argv[2], out.size)
    else:
        main(sys.argv[1], sys.argv[2])
