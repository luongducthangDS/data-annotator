"""Chọn 300 ảnh đường phố từ COCO 2017 để gán nhãn lại từ đầu theo guideline.md.

    python prepare_images.py path/to/instances_train2017.json [instances_val2017.json ...]

Chỉ lấy ảnh có giấy phép CC BY hoặc CC BY-SA. Nhãn COCO KHÔNG được đưa vào CVAT:
chỉ dùng để chọn ảnh có cảnh giao thông. Ghi images/ (không commit) và images.csv (nguồn + giấy phép).
"""
import csv
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

N, SEED = 300, 42
LICENSES = {4: "CC BY 2.0", 5: "CC BY-SA 2.0"}
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
RELEVANT = VEHICLES | {"person", "traffic light"}
HERE = Path(__file__).parent


def candidates(path):
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_image = {}
    for a in coco["annotations"]:
        name = names[a["category_id"]]
        if name in RELEVANT:
            per_image.setdefault(a["image_id"], []).append((name, a["iscrowd"]))
    for img in coco["images"]:
        objs = per_image.get(img["id"], [])
        kinds = {n for n, _ in objs}
        # cảnh đường phố: có ít nhất 2 loại xe, hoặc có xe và đèn tín hiệu; 4-20 vật thể để gán được trong vài phút
        street = len(kinds & VEHICLES) >= 2 or (kinds & VEHICLES and "traffic light" in kinds)
        if img["license"] in LICENSES and street and 4 <= len(objs) <= 20 and not any(c for _, c in objs):
            yield img


def main():
    pool = sorted((i for p in sys.argv[1:] for i in candidates(p)), key=lambda i: i["id"])
    assert len(pool) >= N, f"chỉ có {len(pool)} ảnh đạt điều kiện"
    picked = sorted(random.Random(SEED).sample(pool, N), key=lambda i: i["id"])
    out = HERE / "images"
    out.mkdir(exist_ok=True)

    def fetch(img):
        dst = out / img["file_name"]
        if not dst.exists():
            urllib.request.urlretrieve(img["coco_url"], dst)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(fetch, picked))
    with open(HERE / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "width", "height", "license", "flickr_url", "coco_url"])
        for i in picked:
            w.writerow([i["file_name"], i["width"], i["height"], LICENSES[i["license"]], i["flickr_url"], i["coco_url"]])
    print(f"{len(pool)} ảnh đạt điều kiện, đã chọn {N} -> {out}")


if __name__ == "__main__":
    main()
