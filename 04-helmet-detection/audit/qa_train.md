# QA report — `train_coco.json`

- Ảnh: **1001** | Box: **2819** | Lỗi: **32**
- Box có ít nhất 1 lỗi: **32** (1.1% tổng box)

## Lỗi theo loại

| Loại | Số lượng |
|---|---|
| `too_small` | 31 |
| `duplicate` | 1 |

## Chi tiết (cần sửa)

| Loại | image_id | ann_id | Chi tiết |
|---|---|---|---|
| too_small | 166 | 425 | 5x13px < 10px |
| too_small | 166 | 428 | 8x10px < 10px |
| too_small | 166 | 430 | 5x13px < 10px |
| too_small | 166 | 433 | 10x9px < 10px |
| too_small | 166 | 438 | 12x9px < 10px |
| too_small | 166 | 446 | 9x17px < 10px |
| too_small | 314 | 786 | 8x12px < 10px |
| too_small | 314 | 789 | 9x15px < 10px |
| too_small | 351 | 865 | 10x13px < 10px |
| duplicate | 352 | 869 | trùng với ann 868 |
| too_small | 497 | 1262 | 10x9px < 10px |
| too_small | 497 | 1263 | 8x12px < 10px |
| too_small | 502 | 1299 | 9x17px < 10px |
| too_small | 502 | 1304 | 17x9px < 10px |
| too_small | 502 | 1313 | 10x15px < 10px |
| too_small | 502 | 1319 | 12x10px < 10px |
| too_small | 502 | 1335 | 8x15px < 10px |
| too_small | 502 | 1338 | 5x12px < 10px |
| too_small | 502 | 1340 | 12x8px < 10px |
| too_small | 502 | 1346 | 10x15px < 10px |
| too_small | 502 | 1347 | 8x12px < 10px |
| too_small | 502 | 1351 | 11x9px < 10px |
| too_small | 502 | 1354 | 9x12px < 10px |
| too_small | 502 | 1356 | 10x13px < 10px |
| too_small | 502 | 1359 | 11x9px < 10px |
| too_small | 550 | 1505 | 9x15px < 10px |
| too_small | 746 | 2113 | 9x35px < 10px |
| too_small | 800 | 2302 | 10x10px < 10px |
| too_small | 926 | 2591 | 8x9px < 10px |
| too_small | 926 | 2601 | 4x8px < 10px |
| too_small | 926 | 2603 | 8x12px < 10px |
| too_small | 981 | 2744 | 10x10px < 10px |
