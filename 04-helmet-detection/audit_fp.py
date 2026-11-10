"""Review "false positive" của model trên test set: là lỗi model thật hay là nhãn bị thiếu?

1) python audit_fp.py extract --work work --model work/runs/clean/weights/best.pt
   -> work/audit/candidates.json + lưới ảnh crop work/audit/fp_XX.jpg để review bằng mắt
2) Điền quyết định vào work/audit/decisions.csv (id,verdict) với verdict:
   missing = model đúng, nhãn thiếu | wrong = model sai | unsure
3) python audit_fp.py apply --work work
   -> work/gold/test/labels (nhãn gốc + box 'missing' đã duyệt) + work/gold.yaml + thống kê
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = ["helmet", "head", "motorcycle"]
CONF, IOU = 0.4, 0.5
TILE = 260


def xywhn_to_xyxy(c, W, H):
    cx, cy, w, h = c
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0


def extract(work, model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    test = work / "test"
    out = work / "audit"
    out.mkdir(exist_ok=True)
    cands = []
    for img in sorted((test / "images").glob("*.jpg")):
        im = Image.open(img).convert("RGB")
        W, H = im.size
        gt = [(int(l.split()[0]), xywhn_to_xyxy(map(float, l.split()[1:]), W, H))
              for l in (test / "labels" / (img.stem + ".txt")).read_text().splitlines()]
        r = model.predict(str(img), imgsz=320, conf=CONF, device="cpu", verbose=False)[0]
        for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            cls = int(cls)
            if any(g_c == cls and iou(box, g_b) >= IOU for g_c, g_b in gt):
                continue  # khớp nhãn -> TP, không cần review
            cands.append({"id": len(cands), "image": img.name, "cls": cls, "conf": round(conf, 3),
                          "xyxy": [round(v, 1) for v in box], "W": W, "H": H})
    (out / "candidates.json").write_text(json.dumps(cands, indent=1))
    # lưới 4x4 crop có ngữ cảnh, box đỏ = dự đoán, nhãn = id/class/conf
    for g in range(0, len(cands), 16):
        grid = Image.new("RGB", (TILE * 4, TILE * 4), "white")
        d = ImageDraw.Draw(grid)
        for k, c in enumerate(cands[g:g + 16]):
            im = Image.open(test / "images" / c["image"]).convert("RGB")
            x1, y1, x2, y2 = c["xyxy"]
            pad = max(x2 - x1, y2 - y1) * 0.6
            crop_box = [max(0, x1 - pad), max(0, y1 - pad), min(c["W"], x2 + pad), min(c["H"], y2 + pad)]
            crop = im.crop(crop_box)
            ImageDraw.Draw(crop).rectangle([x1 - crop_box[0], y1 - crop_box[1], x2 - crop_box[0], y2 - crop_box[1]],
                                           outline="red", width=max(2, int(crop.width / 80)))
            crop.thumbnail((TILE, TILE - 20))
            x, y = (k % 4) * TILE, (k // 4) * TILE
            grid.paste(crop, (x, y + 20))
            d.text((x + 3, y + 3), f"{c['id']} {NAMES[c['cls']]} {c['conf']:.2f}", fill="black")
        grid.save(out / f"fp_{g // 16:02d}.jpg", quality=85)
    print(f"{len(cands)} candidates, {(len(cands) + 15) // 16} grids")


def apply(work):
    cands = {c["id"]: c for c in json.loads((work / "audit" / "candidates.json").read_text())}
    with open(work / "audit" / "decisions.csv", encoding="utf-8") as f:
        dec = {int(r["id"]): r["verdict"].strip() for r in csv.DictReader(f)}
    gold = work / "gold" / "test"
    (gold / "labels").mkdir(parents=True, exist_ok=True)
    # Thư mục thật + symlink từng file: ultralytics resolve() đường dẫn thư mục, nếu symlink cả thư mục
    # nó sẽ quay về test/images và đọc nhầm test/labels gốc
    (gold / "images").mkdir(exist_ok=True)
    for img in (work / "test" / "images").glob("*.jpg"):
        if not (gold / "images" / img.name).exists():
            (gold / "images" / img.name).symlink_to(img.resolve())
    added = Counter()
    for lab in (work / "test" / "labels").glob("*.txt"):
        lines = lab.read_text().splitlines()
        for c in cands.values():
            if c["image"][:-4] == lab.stem and dec.get(c["id"]) == "missing":
                x1, y1, x2, y2 = c["xyxy"]
                W, H = c["W"], c["H"]
                lines.append(f"{c['cls']} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                added[NAMES[c["cls"]]] += 1
        (gold / "labels" / lab.name).write_text("\n".join(lines) + "\n")
    (work / "gold.yaml").write_text(f"path: {work.resolve()}\ntrain: clean/train/images\nval: val/images\n"
                                    f"test: gold/test/images\nnames: {dict(enumerate(NAMES))}\n")
    summary = {"reviewed": len(dec), "verdicts": Counter(dec.values()),
               "verdicts_by_class": Counter(f"{NAMES[cands[i]['cls']]}:{v}" for i, v in dec.items()),
               "boxes_added_to_gold": added}
    (work / "audit" / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "apply"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--model")
    a = ap.parse_args()
    extract(Path(a.work), a.model) if a.cmd == "extract" else apply(Path(a.work))
