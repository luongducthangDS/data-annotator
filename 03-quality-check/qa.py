"""Kiểm tra chất lượng annotation COCO trước khi submit + đo độ đồng thuận giữa 2 người gán.

    python qa.py batch.json                       # QA 1 file, in report
    python qa.py batch.json --out report.md       # ghi report ra file
    python qa.py mine.json --compare other.json   # thêm phần đồng thuận (IAA)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MIN_SIDE = 10          # guideline mục 2.3
DUP_IOU = 0.9          # 2 box cùng class chồng >90% = gần như chắc chắn vẽ trùng
MATCH_IOU = 0.5        # ngưỡng khớp box khi so 2 người gán (chuẩn PASCAL VOC)
OBJ_ATTRS = {"occluded": {"none", "partial", "heavy"}, "truncated": {"true", "false"}}
IMG_ATTRS = {"weather": {"clear", "cloudy", "rain", "fog"}, "time_of_day": {"day", "dusk", "night"}}


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load(path):
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        attrs = ann.get("attributes", {})
        if "occlusion" in attrs:  # guideline v1.1: CVAT export dùng `occlusion`, vì `occluded` là cờ bool có sẵn của CVAT
            attrs["occluded"] = attrs["occlusion"]
        by_image[ann["image_id"]].append(ann)
    return coco, by_image


def check(coco, by_image):
    """Trả về list (loại lỗi, image_id, ann_id, chi tiết)."""
    issues = []
    images = {i["id"]: i for i in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    for img_id, img in images.items():
        W, H = img["width"], img["height"]
        anns = by_image.get(img_id, [])
        if not anns:
            issues.append(("empty_image", img_id, None, "ảnh chưa có box — xác nhận là ảnh nền thật hay bị bỏ sót"))
        if "attributes" in img:  # CVAT chỉ xuất thuộc tính ảnh khi có khai báo tag
            for k, allowed in IMG_ATTRS.items():
                v = str(img["attributes"].get(k, "")).lower()
                if v not in allowed:
                    issues.append(("bad_image_attr", img_id, None, f"{k}={v or '∅'}"))
        for ann in anns:
            x, y, w, h = ann["bbox"]
            aid = ann["id"]
            if ann["category_id"] not in cat_ids:
                issues.append(("unknown_class", img_id, aid, f"category_id={ann['category_id']}"))
            if w <= 0 or h <= 0:
                issues.append(("zero_area", img_id, aid, f"w={w}, h={h}"))
                continue
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:  # ±1 px dung sai làm tròn khi export
                issues.append(("out_of_bounds", img_id, aid, f"bbox={[round(v, 1) for v in ann['bbox']]} ảnh {W}x{H}"))
            if w < MIN_SIDE or h < MIN_SIDE:
                issues.append(("too_small", img_id, aid, f"{w:.0f}x{h:.0f}px < {MIN_SIDE}px"))
            attrs = ann.get("attributes", {})
            for k, allowed in OBJ_ATTRS.items():
                v = str(attrs.get(k, "")).lower()
                if v not in allowed:
                    issues.append(("bad_object_attr", img_id, aid, f"{k}={v or '∅'}"))
        # ponytail: O(n²) mỗi ảnh, ổn với vài trăm box/ảnh
        for i, a in enumerate(anns):
            for b in anns[i + 1:]:
                if a["category_id"] == b["category_id"] and iou(a["bbox"], b["bbox"]) > DUP_IOU:
                    issues.append(("duplicate", img_id, b["id"], f"trùng với ann {a['id']}"))
    return issues


def match(ref_anns, other_anns):
    """Ghép box greedy theo IoU giảm dần, bỏ qua class. Trả về list (ref, other, iou)."""
    pairs = sorted(((iou(r["bbox"], o["bbox"]), i, j) for i, r in enumerate(ref_anns)
                    for j, o in enumerate(other_anns)), reverse=True)
    used_r, used_o, out = set(), set(), []
    for v, i, j in pairs:
        if v < MATCH_IOU:
            break
        if i not in used_r and j not in used_o:
            used_r.add(i), used_o.add(j)
            out.append((ref_anns[i], other_anns[j], v))
    return out


def agreement(ref, ref_by, other, other_by):
    names = {c["id"]: c["name"] for c in ref["categories"]}
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion, ious, occ_same, occ_total = Counter(), [], 0, 0
    shared = {i["id"] for i in ref["images"]} & {i["id"] for i in other["images"]}
    for img_id in shared:
        # box diện tích 0 là lỗi vẽ (check() đã báo), không đưa vào so khớp
        r_anns, o_anns = ([a for a in by.get(img_id, []) if a["bbox"][2] > 0 and a["bbox"][3] > 0]
                          for by in (ref_by, other_by))
        pairs = match(r_anns, o_anns)
        matched_r = {id(r) for r, _, _ in pairs}
        matched_o = {id(o) for _, o, _ in pairs}
        for r, o, v in pairs:
            rc, oc = names.get(r["category_id"], "?"), names.get(o["category_id"], "?")
            if rc == oc:
                tp[rc] += 1
                ious.append(v)
                occ_total += 1
                occ_same += r.get("attributes", {}).get("occluded") == o.get("attributes", {}).get("occluded")
            else:  # cùng vị trí nhưng khác class: tính là FN cho class đúng, FP cho class sai
                confusion[(rc, oc)] += 1
                fn[rc] += 1
                fp[oc] += 1
        for r in r_anns:
            if id(r) not in matched_r:
                fn[names.get(r["category_id"], "?")] += 1
        for o in o_anns:
            if id(o) not in matched_o:
                fp[names.get(o["category_id"], "?")] += 1
    return dict(images=len(shared), tp=tp, fp=fp, fn=fn, confusion=confusion,
                mean_iou=sum(ious) / len(ious) if ious else 0.0,
                occ_agree=occ_same / occ_total if occ_total else 0.0)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def report(path, coco, issues, iaa=None):
    n_ann = len(coco["annotations"])
    bad_anns = {(t, a) for t, _, a, _ in issues if a is not None}
    lines = [f"# QA report — `{Path(path).name}`", "",
             f"- Ảnh: **{len(coco['images'])}** | Box: **{n_ann}** | Lỗi: **{len(issues)}**",
             f"- Box có ít nhất 1 lỗi: **{len({a for _, a in bad_anns})}** "
             f"({len({a for _, a in bad_anns}) / n_ann:.1%} tổng box)" if n_ann else "- Không có box", "",
             "## Lỗi theo loại", "", "| Loại | Số lượng |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in Counter(t for t, *_ in issues).most_common()] or ["| (không có) | 0 |"]
    if issues:
        lines += ["", "## Chi tiết (cần sửa)", "", "| Loại | image_id | ann_id | Chi tiết |", "|---|---|---|---|"]
        lines += [f"| {t} | {i} | {a if a is not None else '-'} | {d} |" for t, i, a, d in issues]
    if iaa:
        tp, fp, fn = (sum(iaa[k].values()) for k in ("tp", "fp", "fn"))
        p, r, f = f1(tp, fp, fn)
        lines += ["", f"## Đồng thuận giữa 2 người gán ({iaa['images']} ảnh chung, IoU ≥ {MATCH_IOU})", "",
                  f"- Precision **{p:.2%}** | Recall **{r:.2%}** | F1 **{f:.2%}**",
                  f"- IoU trung bình box khớp: **{iaa['mean_iou']:.3f}**",
                  f"- Đồng ý thuộc tính `occluded`: **{iaa['occ_agree']:.2%}**", "",
                  "| Class | TP | FP | FN | F1 |", "|---|---|---|---|---|"]
        for c in sorted(set(iaa["tp"]) | set(iaa["fp"]) | set(iaa["fn"])):
            lines.append(f"| {c} | {iaa['tp'][c]} | {iaa['fp'][c]} | {iaa['fn'][c]} | {f1(iaa['tp'][c], iaa['fp'][c], iaa['fn'][c])[2]:.2%} |")
        if iaa["confusion"]:
            lines += ["", "**Nhầm class (cùng vị trí, khác nhãn)** → bổ sung edge case vào guideline:", ""]
            lines += [f"- `{a}` ↔ `{b}`: {n} lần" for (a, b), n in iaa["confusion"].most_common()]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("coco_json")
    p.add_argument("--compare", help="file COCO của người gán thứ 2 (cùng image_id)")
    p.add_argument("--out", help="ghi report markdown ra file")
    a = p.parse_args()
    coco, by_image = load(a.coco_json)
    issues = check(coco, by_image)
    iaa = agreement(coco, by_image, *load(a.compare)) if a.compare else None
    text = report(a.coco_json, coco, issues, iaa)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"Đã ghi {a.out} — {len(issues)} lỗi.")
    else:
        print(text)
    raise SystemExit(1 if issues else 0)  # exit 1 để chặn submit khi còn lỗi


if __name__ == "__main__":
    main()
