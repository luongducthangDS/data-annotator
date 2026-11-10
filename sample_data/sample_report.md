# QA report — `annotator_a.json`

- Ảnh: **21** | Box: **115** | Lỗi: **8**
- Box có ít nhất 1 lỗi: **6** (5.2% tổng box)

## Lỗi theo loại

| Loại | Số lượng |
|---|---|
| `duplicate` | 1 |
| `out_of_bounds` | 1 |
| `too_small` | 1 |
| `unknown_class` | 1 |
| `zero_area` | 1 |
| `bad_object_attr` | 1 |
| `empty_image` | 1 |
| `bad_image_attr` | 1 |

## Chi tiết (cần sửa)

| Loại | image_id | ann_id | Chi tiết |
|---|---|---|---|
| duplicate | 1 | 110 | trùng với ann 1 |
| out_of_bounds | 2 | 111 | bbox=[1250, 600, 80, 60] ảnh 1280x720 |
| too_small | 3 | 112 | 6x8px < 10px |
| unknown_class | 4 | 113 | category_id=9 |
| zero_area | 5 | 114 | w=0, h=40 |
| bad_object_attr | 6 | 115 | occluded=∅ |
| empty_image | 21 | - | ảnh chưa có box — xác nhận là ảnh nền thật hay bị bỏ sót |
| bad_image_attr | 21 | - | weather=sunny |

## Đồng thuận giữa 2 người gán (21 ảnh chung, IoU ≥ 0.5)

- Precision **91.51%** | Recall **85.09%** | F1 **88.18%**
- IoU trung bình box khớp: **0.888**
- Đồng ý thuộc tính `occluded`: **90.72%**

| Class | TP | FP | FN | F1 |
|---|---|---|---|---|
| ? | 0 | 0 | 1 | 0.00% |
| bicycle | 6 | 0 | 1 | 92.31% |
| bus | 9 | 0 | 1 | 94.74% |
| car | 14 | 0 | 5 | 84.85% |
| motorbike | 45 | 3 | 5 | 91.84% |
| pedestrian | 13 | 6 | 2 | 76.47% |
| traffic_light | 7 | 0 | 2 | 87.50% |
| truck | 3 | 0 | 0 | 100.00% |

**Nhầm class (cùng vị trí, khác nhãn)** → bổ sung edge case vào guideline:

- `car` ↔ `motorbike`: 2 lần
- `bus` ↔ `motorbike`: 1 lần
