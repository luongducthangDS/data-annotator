# Guideline gán nhãn — Giao thông đô thị (v1.1)

Mục tiêu: dataset object detection + thuộc tính cho ảnh đường phố (300 ảnh COCO 2017, xem `images.csv`).
Công cụ: CVAT (cvat.ai, bản free) hoặc Label Studio. Export: COCO 1.0.

## 1. Danh sách class

| ID | Class | Gồm | Không gồm |
|----|-------|-----|-----------|
| 1 | `motorbike` | xe máy, xe tay ga, xe điện 2 bánh **kèm người lái** (box bao cả người + xe) | xe đạp |
| 2 | `car` | ô tô con, taxi, SUV, bán tải | xe khách ≥16 chỗ |
| 3 | `bus` | xe buýt, xe khách | |
| 4 | `truck` | xe tải, container, xe ben | bán tải (→ car) |
| 5 | `bicycle` | xe đạp, xe đạp điện có bàn đạp | |
| 6 | `pedestrian` | người đi bộ, người đứng | người đang **ngồi trên xe** (thuộc box xe) |
| 7 | `traffic_light` | đèn tín hiệu (cả cụm) | đèn đường chiếu sáng |

## 2. Quy tắc vẽ bounding box

1. Box ôm **sát** phần nhìn thấy của vật thể, sai số ≤ 2 px mỗi cạnh.
2. Vật thể bị che: chỉ vẽ phần **nhìn thấy**, không đoán phần bị che.
3. Bỏ qua vật thể nhỏ hơn **10×10 px** (không gán nhãn).
4. Đám đông/dòng xe dày không tách được → **không** vẽ 1 box gộp; vẽ từng chiếc tách được, phần còn lại bỏ qua.
5. Ảnh phản chiếu (kính, gương), poster, biển quảng cáo → **không** gán nhãn.
6. Vật thể cắt ở mép ảnh → vẫn gán nhãn, đánh `truncated = true`.

## 3. Thuộc tính (bắt buộc)

**Theo object**

| Thuộc tính | Giá trị | Quy tắc |
|------------|---------|---------|
| `occlusion` | `none` / `partial` / `heavy` | partial: bị che < 50%; heavy: ≥ 50% |
| `truncated` | `true` / `false` | bị mép ảnh cắt |

**Theo ảnh**

| Thuộc tính | Giá trị |
|------------|---------|
| `weather` | `clear` / `cloudy` / `rain` / `fog` |
| `time_of_day` | `day` / `dusk` / `night` |

## 4. Trường hợp khó (edge cases) — tham khảo trước khi hỏi lead

| Tình huống | Quyết định |
|------------|-----------|
| Xe máy chở 2–3 người | 1 box `motorbike` duy nhất |
| Người dắt xe máy (không ngồi) | 1 `pedestrian` + 1 `motorbike` riêng |
| Xe ba gác, xe lôi | `truck` nếu chở hàng, còn lại bỏ qua → ghi vào `issues.md` |
| Đèn giao thông quay ngang/ngược | vẫn gán `traffic_light` |
| Ảnh mờ không xác định được class | không gán, ghi image_id vào `issues.md` |

## 5. Checklist trước khi submit mỗi batch

- [ ] Mọi object đều có `occlusion`, `truncated`
- [ ] Mọi ảnh đều có `weather`, `time_of_day`
- [ ] Không có box thừa/trùng (chạy `03-quality-check/qa.py`)
- [ ] Điền thời gian vào `labeling_log.csv`

## Changelog
- v1.0 — bản đầu.
- v1.1 — đổi thuộc tính `occluded` thành `occlusion` vì CVAT đã có sẵn cờ `occluded` kiểu đúng/sai, trùng tên sẽ ghi đè khi export COCO. Đổi nguồn ảnh sang 300 ảnh đường phố COCO 2017 (CC BY, CC BY-SA).
