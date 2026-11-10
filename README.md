# Data Annotator — Portfolio

Bộ dự án chứng minh năng lực cho vị trí **Nhân viên gán nhãn / xử lý dữ liệu hình ảnh** (2D, bounding box, thuộc tính, export format, QA).

## Project chính: `04-helmet-detection/`

Audit 1.429 ảnh của một dataset "helmet detection" công khai (Open Images), train YOLOv8n trên 5 biến thể nhãn, review từng dự đoán sai để sửa test set.

- **0/1.429 ảnh** gán đủ các class cùng lúc, nên dataset không dùng được cho bài toán người đi xe máy không đội mũ.
- **37%** "false positive" của model thực ra là **nhãn bị thiếu**. Sửa test set thì mAP50 tăng từ 0.563 lên 0.602.
- **Box lệch tới 20% kích thước làm mAP50 giảm 11.2 điểm**, nhầm class 30% giảm 6.2, bỏ sót 40% box giảm 4.6 (1 seed; xem phần giới hạn).

Câu chuyện phỏng vấn trong 3 câu: *"Tôi audit một dataset công khai trước khi dùng và phát hiện mỗi ảnh chỉ được gán một class, nên model bị phạt khi tìm ra đúng vật thể. Tôi review từng dự đoán sai: 37% là do nhãn thiếu chứ không phải lỗi model. Thí nghiệm nhiễu nhãn cho thấy vẽ box ẩu hại model gấp khoảng 2.5 lần bỏ sót box, nên khi gán tôi ưu tiên độ chính xác của box và định nghĩa class rõ ràng (mũ bảo hiểm khác mũ lưỡi trai, nón lá)."*

`01-traffic-labeling/` là phần gán tay trên CVAT theo guideline giao thông. Phần này **chưa làm**; nên làm khi có thời gian, vì bài test tuyển dụng sẽ kiểm tra thao tác CVAT.

## Map yêu cầu tuyển dụng → bằng chứng

| Yêu cầu trong JD | Bằng chứng trong repo |
|------------------|----------------------|
| Đánh dấu, khoanh vùng đối tượng theo quy tắc | `01-traffic-labeling/` — guideline + kế hoạch gán 300 ảnh trên CVAT (**chưa làm**) |
| Điền thông tin đối tượng (thời tiết, phương tiện, …) | Thuộc tính `weather`, `time_of_day`, `occluded`, `truncated` — `guideline.md` |
| Làm theo guideline được đào tạo | Tự viết guideline có edge case + changelog |
| Trích xuất ra định dạng theo yêu cầu | `02-format-converter/` — COCO → YOLO / Pascal VOC |
| Kiểm tra chất lượng trước khi submit | `03-quality-check/qa.py` — 8 loại lỗi, exit 1 nếu còn lỗi; `04/audit/` — review tay 45 ảnh + 93 dự đoán sai |
| Hiểu nhãn dùng để làm gì cho AI | `04-helmet-detection/` — đo lỗi gán nhãn làm model giảm bao nhiêu |
| Báo cáo tiến độ, hiệu quả | `labeling_log.csv` (object/giờ) + QA report + độ đồng thuận 2 người |
| Kinh nghiệm 2D/3D | Box + polygon + video tracking (mở rộng trong `01/README.md`) |

## Chạy thử (Python 3.9+, không cần thư viện ngoài)

```bash
cd sample_data && python make_sample.py && cd ..
python 03-quality-check/qa.py sample_data/annotator_a.json --compare sample_data/annotator_b.json --out sample_data/sample_report.md
python 02-format-converter/convert.py sample_data/annotator_a.json --to yolo --out out_yolo
pip install pytest && python -m pytest -q tests
```

`sample_data/` là dữ liệu **giả lập** có cài sẵn lỗi để demo công cụ — không phải kết quả gán nhãn.

## QA bắt những lỗi gì

| Lỗi | Nghĩa |
|-----|------|
| `duplicate` | 2 box cùng class chồng > 90% |
| `out_of_bounds` | box tràn khỏi ảnh |
| `too_small` | cạnh < 10 px (guideline yêu cầu bỏ) |
| `zero_area` | box rộng/cao = 0 |
| `unknown_class` | class không có trong danh sách |
| `bad_object_attr` / `bad_image_attr` | thiếu hoặc sai giá trị thuộc tính |
| `empty_image` | ảnh chưa gán box nào |

Độ đồng thuận (IAA): ghép box theo IoU ≥ 0.5 → Precision / Recall / F1 theo class, IoU trung bình, % đồng ý `occluded`, và danh sách cặp class hay nhầm để cập nhật guideline.
