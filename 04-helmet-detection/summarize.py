"""Gộp kết quả nhiều seed thành bảng markdown: mean ± std, chênh lệch so với clean.

    python summarize.py work/results_gpu --out work/results_gpu/summary.md
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ORDER = ["clean", "drop20", "drop40", "flip30", "jitter20"]


def fmt(xs):
    return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in Path(a.results_dir).glob("*__*.json")]
    by = defaultdict(list)
    for r in rows:
        by[(r["test_set"], r["variant"])].append(r)

    lines = []
    for test_set in ("original", "gold"):
        variants = [v for v in ORDER if (test_set, v) in by]
        if not variants:
            continue
        base = st.mean(r["mAP50"] for r in by[(test_set, "clean")]) if (test_set, "clean") in by else None
        n = len(by[(test_set, variants[0])])
        lines += [f"## Test set: {test_set} (n = {n} seed/biến thể)", "",
                  "| Nhãn train | mAP50 | Δ mAP50 vs clean | mAP50-95 | helmet | head | motorcycle |",
                  "|---|---|---|---|---|---|---|"]
        for v in variants:
            rs = by[(test_set, v)]
            m50 = [r["mAP50"] for r in rs]
            delta = "—" if v == "clean" or base is None else f"{(st.mean(m50) - base) * 100:+.1f}"
            pc = {c: [r["per_class_mAP50"].get(c, 0) for r in rs] for c in ("helmet", "head", "motorcycle")}
            lines.append(f"| {v} | {fmt(m50)} | {delta} | {fmt([r['mAP50_95'] for r in rs])} | "
                         f"{st.mean(pc['helmet']):.3f} | {st.mean(pc['head']):.3f} | {st.mean(pc['motorcycle']):.3f} |")
        lines.append("")
    text = "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
