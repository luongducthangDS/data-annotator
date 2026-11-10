# 01 — Gán nhãn 300 ảnh giao thông đô thị (bounding box + thuộc tính)

> Phần này **phải tự làm tay** — đây chính là bằng chứng kỹ năng mà nhà tuyển dụng cần.

## Kết quả cần có (điền sau khi làm)

| Chỉ số | Giá trị |
|--------|---------|
| Số ảnh | 300 |
| Số object | … |
| Tốc độ trung bình | … object/giờ |
| Tỉ lệ lỗi QA lần 1 → sau sửa | …% → …% |
| Độ đồng thuận với người thứ 2 (F1 @ IoU 0.5) | … |

## Các bước

1. **Dữ liệu** (đã có): 300 ảnh đường phố COCO 2017, giấy phép CC BY hoặc CC BY-SA, chọn bằng `prepare_images.py` (seed 42).
   Nguồn và giấy phép từng ảnh ở `images.csv`. Nhãn COCO không được nạp vào CVAT; mọi box đều gán tay từ đầu.
   Ảnh nằm ở `images/` và `images_300.zip` (không commit; chạy lại `prepare_images.py` để tải).
2. **Tạo task trên [app.cvat.ai](https://app.cvat.ai)** (bản free): Tasks → Create a new task → ở mục Labels chọn tab **Raw**,
   dán nội dung `cvat_labels.json` → Done. Kéo `images_300.zip` vào Select files. Advanced configuration → Segment size = 50
   (ra 6 job, mỗi job là một batch). Nhãn `scene` là tag cho cả ảnh (`weather`, `time_of_day`); 7 nhãn còn lại là box.
3. **Gán nhãn** theo `guideline.md`, chia batch 50 ảnh. Sau mỗi batch ghi vào `labeling_log.csv`.
4. **Export**: Export task dataset, lấy cả hai định dạng **COCO 1.0** (box) và **CVAT for images 1.1** (có tag `scene`) → `annotations/`.
5. **QA**: `python ../03-quality-check/qa.py annotations/batch_xx.json --out reports/batch_xx.md` → sửa lỗi → chạy lại.
6. **Đồng thuận (IAA)**: nhờ 1 người bạn gán lại 30 ảnh theo cùng guideline →
   `python ../03-quality-check/qa.py mine.json --compare friend.json`.
   Chỗ lệch nhiều → bổ sung edge case vào guideline (tăng version).
7. **Export format khác**: `python ../02-format-converter/convert.py annotations/all.json --to yolo --out yolo/`.
8. Chụp 4–6 ảnh màn hình CVAT (trước/sau, edge case) đưa vào `screenshots/`.

## Mở rộng (nếu còn thời gian, đúng phần "ưu tiên 2D/3D")
- Polygon/segmentation cho 50 ảnh (class `pedestrian`, `motorbike`).
- Video tracking 1–2 clip ngắn trên CVAT (interpolation giữa keyframe).
- 3D: thử gán cuboid trên vài frame point cloud mẫu của CVAT/Segments.ai — chỉ cần ghi "đã làm quen", đừng phóng đại.
