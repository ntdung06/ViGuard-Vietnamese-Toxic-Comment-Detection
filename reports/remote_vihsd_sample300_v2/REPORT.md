# Kết quả đánh giá mô hình API

Model cấu hình: `gpt-4o-mini`; nguồn API: `https://cheapkeyai.shop/v1`.
Tập dữ liệu do người chạy khai báo: ViHSD; split: test.
Số mẫu đánh giá: 300/6680; chọn mẫu: random sample, seed=42.
SHA256 file dữ liệu: `ba8ae9b62c40caebcbe8af9a09590296bca286e83e22dd639de31a62ed16f5a5`.

- Macro-F1: 0.6055
- Accuracy: 0.8067

| Nhãn | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CLEAN | 0.9469 | 0.8664 | 0.9049 | 247 |
| OFFENSIVE | 0.2600 | 0.5652 | 0.3562 | 23 |
| HATE | 0.6250 | 0.5000 | 0.5556 | 30 |

Xem confusion_matrix.csv để đọc lỗi theo lớp. Báo cáo này đánh giá API bên thứ ba, không phải mô hình BiLSTM/PhoBERT do nhóm tự huấn luyện. Nếu chỉ chọn một phần dữ liệu, phải ghi rõ đây là đánh giá trên mẫu. Dữ liệu tiền huấn luyện của API không được biết; không thể loại trừ khả năng đã gặp dữ liệu benchmark.

Chi tiết nguồn dữ liệu, hướng dẫn nhãn đã sửa và lịch sử timeout/chia nhỏ batch: xem `../VIHSD_RESULTS.md`.
