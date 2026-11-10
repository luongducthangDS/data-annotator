"""Chuyển annotation COCO (export từ CVAT/Label Studio) sang YOLO hoặc Pascal VOC.

    python convert.py annotations.json --to yolo --out out_yolo/
    python convert.py annotations.json --to voc  --out out_voc/
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


def load_coco(path):
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return images, cats, by_image


def clip_box(x, y, w, h, W, H):
    """Cắt box về trong khung ảnh; trả None nếu box rỗng sau khi cắt."""
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def to_yolo(coco_path, out_dir):
    images, cats, by_image = load_coco(coco_path)
    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    # YOLO dùng index 0..n-1, COCO dùng id bất kỳ -> map lại
    idx = {c["id"]: i for i, c in enumerate(cats)}
    (out / "classes.txt").write_text("\n".join(c["name"] for c in cats) + "\n", encoding="utf-8")
    skipped = 0
    for img_id, img in images.items():
        W, H = img["width"], img["height"]
        lines = []
        for ann in by_image.get(img_id, []):
            box = clip_box(*ann["bbox"], W, H)
            if box is None or ann["category_id"] not in idx:
                skipped += 1
                continue
            x1, y1, x2, y2 = box
            lines.append(f"{idx[ann['category_id']]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        # ảnh không có object vẫn cần file rỗng để YOLO hiểu là "background"
        (out / "labels" / (Path(img["file_name"]).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    return len(images), skipped


def to_voc(coco_path, out_dir):
    images, cats, by_image = load_coco(coco_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in cats}
    skipped = 0
    for img_id, img in images.items():
        W, H = img["width"], img["height"]
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = img["file_name"]
        size = ET.SubElement(root, "size")
        for k, v in (("width", W), ("height", H), ("depth", 3)):
            ET.SubElement(size, k).text = str(v)
        for ann in by_image.get(img_id, []):
            box = clip_box(*ann["bbox"], W, H)
            if box is None or ann["category_id"] not in names:
                skipped += 1
                continue
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = names[ann["category_id"]]
            attrs = ann.get("attributes", {})
            ET.SubElement(obj, "truncated").text = "1" if str(attrs.get("truncated")).lower() == "true" else "0"
            ET.SubElement(obj, "difficult").text = "1" if attrs.get("occluded") == "heavy" else "0"
            bb = ET.SubElement(obj, "bndbox")
            # VOC dùng pixel 1-based, số nguyên
            for k, v in zip(("xmin", "ymin", "xmax", "ymax"), box):
                ET.SubElement(bb, k).text = str(max(1, round(v)))
        ET.indent(root)
        ET.ElementTree(root).write(out / (Path(img["file_name"]).stem + ".xml"), encoding="utf-8")
    return len(images), skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("coco_json")
    p.add_argument("--to", choices=["yolo", "voc"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n, skipped = (to_yolo if a.to == "yolo" else to_voc)(a.coco_json, a.out)
    print(f"Đã xuất {n} ảnh sang {a.to.upper()} tại {a.out}. Bỏ qua {skipped} box lỗi (rỗng/sai class) — chạy qa.py để xem chi tiết.")


if __name__ == "__main__":
    main()
