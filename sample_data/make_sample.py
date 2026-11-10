"""Sinh 2 file COCO giả lập (2 người gán cùng 20 ảnh) có cài sẵn lỗi, để demo convert.py và qa.py.
Không thay thế dữ liệu thật ở 01-traffic-labeling.

    python make_sample.py   # tạo annotator_a.json, annotator_b.json
"""
import json
import random
from pathlib import Path

CATS = ["motorbike", "car", "bus", "truck", "bicycle", "pedestrian", "traffic_light"]
HERE = Path(__file__).parent


def build(seed=7):
    rng = random.Random(seed)
    images, a_anns, b_anns = [], [], []
    for img_id in range(1, 21):
        images.append({"id": img_id, "file_name": f"hn_{img_id:04d}.jpg", "width": 1280, "height": 720,
                       "attributes": {"weather": rng.choice(["clear", "cloudy", "rain"]), "time_of_day": "day"}})
        for _ in range(rng.randint(3, 8)):
            w, h = rng.randint(30, 200), rng.randint(30, 200)
            box = [rng.randint(0, 1280 - w), rng.randint(0, 720 - h), w, h]
            cat = rng.choices(range(1, 8), weights=[8, 3, 1, 1, 1, 3, 1])[0]
            attrs = {"occluded": rng.choice(["none", "none", "partial", "heavy"]), "truncated": "false"}
            a_anns.append({"image_id": img_id, "category_id": cat, "bbox": box, "attributes": attrs})
            # người B: lệch vài px, đôi khi bỏ sót / nhầm class / khác occluded
            r = rng.random()
            if r < 0.08:
                continue
            b_cat = 1 if (r < 0.14 and cat == 5) or (r < 0.12 and cat != 1) else cat
            jitter = [v + rng.randint(-4, 4) for v in box]
            b_attrs = dict(attrs, occluded=rng.choice(["none", "partial"]) if r > 0.9 else attrs["occluded"])
            b_anns.append({"image_id": img_id, "category_id": b_cat, "bbox": jitter, "attributes": b_attrs})
        if rng.random() < 0.3:  # người B vẽ thừa
            b_anns.append({"image_id": img_id, "category_id": 6, "bbox": [rng.randint(0, 1100), rng.randint(0, 600), 40, 90],
                           "attributes": {"occluded": "none", "truncated": "false"}})

    # cài lỗi vào file A để QA bắt được
    a_anns.append(dict(a_anns[0], bbox=[v + 1 for v in a_anns[0]["bbox"]]))                 # duplicate
    a_anns.append({"image_id": 2, "category_id": 2, "bbox": [1250, 600, 80, 60], "attributes": {"occluded": "none", "truncated": "true"}})  # out_of_bounds
    a_anns.append({"image_id": 3, "category_id": 7, "bbox": [500, 100, 6, 8], "attributes": {"occluded": "none", "truncated": "false"}})   # too_small
    a_anns.append({"image_id": 4, "category_id": 9, "bbox": [100, 100, 50, 50], "attributes": {"occluded": "none", "truncated": "false"}})  # unknown_class
    a_anns.append({"image_id": 5, "category_id": 1, "bbox": [300, 300, 0, 40], "attributes": {"occluded": "none", "truncated": "false"}})  # zero_area
    a_anns.append({"image_id": 6, "category_id": 1, "bbox": [700, 300, 60, 90], "attributes": {"truncated": "false"}})                     # thiếu occluded
    images.append({"id": 21, "file_name": "hn_0021.jpg", "width": 1280, "height": 720,
                   "attributes": {"weather": "sunny", "time_of_day": "day"}})                                                            # empty + bad weather

    def coco(anns):
        return {"images": images, "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CATS)],
                "annotations": [dict(a, id=k + 1, area=a["bbox"][2] * a["bbox"][3], iscrowd=0) for k, a in enumerate(anns)]}

    return coco(a_anns), coco(b_anns)


if __name__ == "__main__":
    a, b = build()
    for name, data in (("annotator_a.json", a), ("annotator_b.json", b)):
        (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"A: {len(a['annotations'])} box, B: {len(b['annotations'])} box, {len(a['images'])} ảnh")
