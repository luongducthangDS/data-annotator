"""Chuẩn bị dữ liệu: đọc nhãn YOLO gốc, audit, chia train/val/test, sinh các biến thể nhiễu nhãn.

    python prepare.py --src <thư mục dataset gốc có images/ labels/> --out work/
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

# Open Images: 0 Helmet, 1 Human head, 2 Motorcycle, 3 Vehicle registration plate
KEEP = {0: 0, 1: 1, 2: 2}          # bỏ biển số: không cần cho bài toán + tránh lộ thông tin
NAMES = ["helmet", "head", "motorcycle"]
SEED = 42


def read_labels(path):
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5 and int(p[0]) in KEEP:
            rows.append([KEEP[int(p[0])], *map(float, p[1:])])
    return rows


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u else 0


def to_coco(items, out_json):
    """Xuất COCO để tái dùng qa.py / convert.py ở thư mục 03, 02."""
    images, anns = [], []
    for i, (img, W, H, rows) in enumerate(items, 1):
        images.append({"id": i, "file_name": img.name, "width": W, "height": H})
        for c, cx, cy, w, h in rows:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c + 1,
                         "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                         "attributes": {"occluded": "none", "truncated": "false"}})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]
    Path(out_json).write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))


def write_split(out, name, items, transform=None):
    for sub in ("images", "labels"):
        (out / name / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for img, _, _, rows in items:
        link = out / name / "images" / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        rows = transform(rows, rng) if transform else rows
        (out / name / "labels" / (img.stem + ".txt")).write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows))


def drop_boxes(p):
    """Mô phỏng annotator bỏ sót: xoá ngẫu nhiên p% box."""
    return lambda rows, rng: [r for r in rows if rng.random() >= p]


def flip_class(p):
    """Mô phỏng nhầm nhãn helmet <-> head (lỗi nghiệp vụ quan trọng nhất của bài toán)."""
    def f(rows, rng):
        return [[1 - r[0], *r[1:]] if r[0] in (0, 1) and rng.random() < p else r for r in rows]
    return f


def jitter_boxes(p):
    """Mô phỏng vẽ box ẩu: dịch/co giãn mỗi cạnh ngẫu nhiên tới p * kích thước box."""
    def f(rows, rng):
        out = []
        for c, cx, cy, w, h in rows:
            cx += rng.uniform(-p, p) * w
            cy += rng.uniform(-p, p) * h
            w *= 1 + rng.uniform(-p, p)
            h *= 1 + rng.uniform(-p, p)
            out.append([c, min(max(cx, 0), 1), min(max(cy, 0), 1), min(w, 1), min(h, 1)])
        return out
    return f


VARIANTS = {
    "clean": None,
    "drop20": drop_boxes(0.2), "drop40": drop_boxes(0.4),
    "flip10": flip_class(0.1), "flip30": flip_class(0.3),
    "jitter20": jitter_boxes(0.2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="work")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    items, stats = [], Counter()
    for img in sorted((src / "images").glob("*.jpg")):
        lab = src / "labels" / (img.stem + ".txt")
        if not lab.exists():
            stats["no_label_file"] += 1
            continue
        rows = read_labels(lab)
        if not rows:
            stats["only_plates_or_empty"] += 1
            continue
        W, H = Image.open(img).size
        for r in rows:
            stats[f"box_{NAMES[r[0]]}"] += 1
        # helmet và head trùng gần như hoàn toàn -> nhãn mâu thuẫn (cùng 1 cái đầu vừa có vừa không mũ)
        hel = [r[1:] for r in rows if r[0] == 0]
        head = [r[1:] for r in rows if r[0] == 1]
        stats["helmet_head_conflict"] += sum(1 for h in hel for d in head if iou_xywh(h, d) > 0.7)
        items.append((img, W, H, rows))

    rng = random.Random(SEED)
    rng.shuffle(items)
    n = len(items)
    test, val, train = items[:int(.15 * n)], items[int(.15 * n):int(.30 * n)], items[int(.30 * n):]
    stats.update(images_used=n, train=len(train), val=len(val), test=len(test))

    write_split(out, "test", test)
    write_split(out, "val", val)
    for v, t in VARIANTS.items():
        write_split(out / v, "train", train, t)
        (out / f"{v}.yaml").write_text(
            f"path: {out.resolve()}\ntrain: {v}/train/images\nval: val/images\ntest: test/images\n"
            f"names: {dict(enumerate(NAMES))}\n")
    to_coco(test, out / "test_coco.json")
    to_coco(train, out / "train_coco.json")
    (out / "split.json").write_text(json.dumps({k: [i[0].name for i in s] for k, s in
                                                (("train", train), ("val", val), ("test", test))}))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
