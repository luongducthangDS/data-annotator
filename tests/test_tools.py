import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
for sub in ("02-format-converter", "03-quality-check", "sample_data"):
    sys.path.insert(0, str(ROOT / sub))

import json  # noqa: E402
import convert  # noqa: E402
import make_sample  # noqa: E402
import qa  # noqa: E402


def write(tmp_path, data, name="a.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_qa_catches_every_planted_error(tmp_path):
    a, _ = make_sample.build()
    coco, by_image = qa.load(write(tmp_path, a))
    found = {t for t, *_ in qa.check(coco, by_image)}
    assert {"duplicate", "out_of_bounds", "too_small", "unknown_class", "zero_area",
            "bad_object_attr", "empty_image", "bad_image_attr"} <= found


def test_clean_file_has_no_issues(tmp_path):
    clean = {"images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
             "categories": [{"id": 1, "name": "car"}],
             "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30],
                              "attributes": {"occluded": "none", "truncated": "false"}}]}
    assert qa.check(*qa.load(write(tmp_path, clean))) == []


def test_agreement_identical_files_is_perfect(tmp_path):
    a, _ = make_sample.build()
    p = write(tmp_path, a)
    res = qa.agreement(*qa.load(p), *qa.load(p))
    assert sum(res["fp"].values()) == 0 and sum(res["fn"].values()) == 0
    assert abs(res["mean_iou"] - 1.0) < 1e-9


def test_agreement_counts_class_confusion(tmp_path):
    base = {"images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}]}
    a = dict(base, annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40]}])
    b = dict(base, annotations=[{"id": 1, "image_id": 1, "category_id": 2, "bbox": [11, 11, 40, 40]}])
    res = qa.agreement(*qa.load(write(tmp_path, a)), *qa.load(write(tmp_path, b, "b.json")))
    assert res["confusion"][("car", "bus")] == 1 and res["fn"]["car"] == 1 and res["fp"]["bus"] == 1


def test_yolo_export_normalized_and_clipped(tmp_path):
    data = {"images": [{"id": 1, "file_name": "img.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 5, "name": "car"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [50, 25, 100, 50]},
                            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [150, 0, 100, 100]}]}  # tràn phải
    n, skipped = convert.to_yolo(write(tmp_path, data), tmp_path / "y")
    rows = (tmp_path / "y/labels/img.txt").read_text().splitlines()
    assert (n, skipped) == (1, 0)
    assert rows[0] == "0 0.500000 0.500000 0.500000 0.500000"
    assert rows[1].split()[3] == "0.250000"  # đã cắt về mép ảnh: 150..200
    assert (tmp_path / "y/classes.txt").read_text().strip() == "car"


def test_voc_export(tmp_path):
    a, _ = make_sample.build()
    n, skipped = convert.to_voc(write(tmp_path, a), tmp_path / "v")
    assert n == 21 and skipped == 2  # unknown_class + zero_area bị bỏ
    root = ET.parse(tmp_path / "v/hn_0001.xml").getroot()
    assert root.find("size/width").text == "1280" and root.findall("object")
