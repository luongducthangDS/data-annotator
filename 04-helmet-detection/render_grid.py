"""Vẽ box lên ảnh và ghép lưới 3x3 để review nhãn bằng mắt.

    python render_grid.py work/test --out work/grids --n 90
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = [(0, 200, 0), (230, 0, 0), (0, 120, 255)]  # helmet xanh lá, head đỏ, motorcycle xanh dương
CELL = 420


def draw(img_path, label_path):
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((CELL, CELL))
    W, H = im.size
    d = ImageDraw.Draw(im)
    for line in label_path.read_text().splitlines():
        c, cx, cy, w, h = line.split()
        c, cx, cy, w, h = int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COLORS[c], width=3)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=90)
    a = ap.parse_args()
    split, out = Path(a.split_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((split / "images").glob("*.jpg"))[:a.n]
    for g in range(0, len(imgs), 9):
        grid = Image.new("RGB", (CELL * 3, CELL * 3 + 0), "white")
        d = ImageDraw.Draw(grid)
        for k, p in enumerate(imgs[g:g + 9]):
            x, y = (k % 3) * CELL, (k // 3) * CELL
            grid.paste(draw(p, split / "labels" / (p.stem + ".txt")), (x, y))
            d.rectangle([x, y, x + 150, y + 22], fill="black")
            d.text((x + 4, y + 4), f"#{g + k} {p.stem[:10]}", fill="yellow")
        grid.save(out / f"grid_{g // 9:02d}.jpg", quality=80)


if __name__ == "__main__":
    main()
