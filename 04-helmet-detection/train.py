"""Fine-tune YOLOv8n cho từng biến thể nhãn train, đánh giá trên cùng 1 test set.

    # CPU, 1 seed (bản đã chạy trong README)
    python train.py --work work --variants clean drop40 --epochs 20 --imgsz 320
    # GPU (Colab), 3 seed, đánh giá luôn trên test gốc + gold
    python train.py --work work --variants clean drop20 --seeds 0 1 2 --epochs 50 --imgsz 640 \
        --device 0 --batch 32 --gold work/gold.yaml --results results_gpu
Kết quả: work/<results>/<run>__<test_set>.json  (run = variant hoặc variant_s<seed>)
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights, data, test_set, run, variant, seed, a, minutes):
    m = YOLO(str(weights)).val(data=data, split="test", imgsz=a.imgsz, batch=a.batch, device=a.device,
                               plots=False, verbose=False, project=str(Path(a.work).resolve() / "val"),
                               name=f"{run}__{test_set}", exist_ok=True)
    names = m.names
    return {"variant": variant, "seed": seed, "test_set": test_set,
            "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
            "per_class_mAP50": {names[i]: round(float(x), 4) for i, x in zip(m.box.ap_class_index, m.box.ap50)},
            "train_minutes": minutes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--variants", nargs="+", default=["clean"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="bỏ trống = 1 lần chạy seed 0, tên run = variant")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gold", default=None, help="yaml test set đã sửa; đánh giá thêm trên set này")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    work = Path(a.work).resolve()
    res_dir = work / a.results
    res_dir.mkdir(exist_ok=True)

    for v in a.variants:
        for seed in (a.seeds or [0]):
            run = f"{v}_s{seed}" if a.seeds else v
            weights = work / "runs" / run / "weights" / "best.pt"
            t0 = time.time()
            if not weights.exists():  # chạy lại notebook sẽ bỏ qua run đã xong
                YOLO("yolov8n.pt").train(data=str(work / f"{v}.yaml"), epochs=a.epochs, imgsz=a.imgsz,
                                         batch=a.batch, workers=2, device=a.device, seed=seed, deterministic=True,
                                         plots=False, project=str(work / "runs"), name=run, exist_ok=True,
                                         verbose=False)
            minutes = round((time.time() - t0) / 60, 1)
            tests = [("original", str(work / f"{v}.yaml"))] + ([("gold", a.gold)] if a.gold else [])
            for test_set, data in tests:
                out = res_dir / f"{run}__{test_set}.json"
                if out.exists():
                    continue
                row = evaluate(weights, data, test_set, run, v, seed, a, minutes)
                out.write_text(json.dumps(row, indent=1))
                print(json.dumps(row))


if __name__ == "__main__":
    main()
