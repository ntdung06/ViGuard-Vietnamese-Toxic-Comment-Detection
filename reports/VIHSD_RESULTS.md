# Kết quả thực nghiệm ViHSD

Ngày chạy: 3 tháng 11 năm 2026. Dữ liệu tải từ bucket người dùng cung cấp `hf://buckets/KHoa121/vihsd-bucket`, dùng lại bản ViHSD. Tập nguồn có 24.048 train, 2.672 dev, 6.680 test. Hash test gốc: `ba8ae9b62c40caebcbe8af9a09590296bca286e83e22dd639de31a62ed16f5a5`.

## Model API đang dùng trên web

Model cấu hình `gpt-4o-mini`, endpoint CheapKeyAI, temperature 0. Chọn ngẫu nhiên không hoàn lại 300 câu trong test gốc, seed 42, trước khi tính chỉ số. Không gửi nhãn thật đến nhà cung cấp. Dùng 247 CLEAN, 23 OFFENSIVE, 30 HATE. Giữ mọi mẫu và hoàn tất 300/300 dự đoán.

- Macro-F1: **0,6055**.
- Accuracy: **80,67%**, tương ứng 242/300 câu đúng.

| Nhãn | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CLEAN | 0.9469 | 0.8664 | 0.9049 | 247 |
| OFFENSIVE | 0.2600 | 0.5652 | 0.3562 | 23 |
| HATE | 0.6250 | 0.5000 | 0.5556 | 30 |

Ma trận nhầm lẫn, hàng là nhãn thật:

| Thật / Dự đoán | CLEAN | OFFENSIVE | HATE |
|---|---:|---:|---:|
| CLEAN | 214 | 25 | 8 |
| OFFENSIVE | 9 | 13 | 1 |
| HATE | 3 | 12 | 15 |

Model nhầm 25 câu CLEAN và 12 câu HATE sang OFFENSIVE. OFFENSIVE precision chỉ 0,26; HATE recall 0,50. Đây là điểm cần cải thiện; 23/30 mẫu của các lớp ít không đủ để khẳng định chất lượng ổn định. Chưa có kết quả trên toàn bộ 6.680 câu test.

Lượt prototype ban đầu dừng do timeout, không có metrics hoàn chỉnh. Hướng dẫn nhãn trong prompt được chỉnh theo Bảng 1 của bài báo gốc, vì HATE của ViHSD bao gồm công kích có đích vào cá nhân. Chỉnh sửa dựa trên tài liệu hướng dẫn, trước khi xem điểm API; không chọn prompt dựa trên kết quả test. Prompt/hash model cuối nằm trong `evaluation_manifest.json`.

Lượt cuối dùng tối đa 3 request đồng thời, timeout 120 giây/request. Sau timeout, checkpoint giữ 290 câu; 10 câu còn lại được gọi riêng, cùng prompt/model. Không bỏ các câu lỗi và không thay mẫu. Cỡ batch 10 rồi 1 được ghi trong manifest; kết quả có thể phụ thuộc ngữ cảnh batch.

## Baseline tự huấn luyện

Pipeline chuẩn hóa Unicode/khoảng trắng, kiểm tra trùng/xung đột, ngăn giao nhau theo khóa basic và teencode. Cùng snapshot dùng cho cả hai nhánh. Sau lọc: 21.287 train, 2.556 dev, 6.567 test. TF-IDF chỉ fit train; chọn mode/C/class_weight bằng dev Macro-F1; test chỉ chấm cấu hình được chọn.

Cấu hình chọn: teencode, TF-IDF word n-gram (1,2), Logistic Regression C=1, class_weight=balanced. Dev Macro-F1 0,6055; test Macro-F1 **0,6189**, accuracy **82,50%**. F1 CLEAN 0,9092; OFFENSIVE 0,4051; HATE 0,5422.

Dummy dự đoán lớp phổ biến nhất đạt accuracy 83,04% nhưng Macro-F1 chỉ 0,3024. Kết quả này cho thấy cần ưu tiên Macro-F1 với phân bố lớp lệch.

Baseline dùng curated split, API dùng mẫu từ test gốc. Không suy ra model nào tốt hơn chỉ bằng hai điểm này. Muốn so trực tiếp phải chốt cùng một tập đánh giá và cùng chính sách dữ liệu, rồi công khai các lần đã sử dụng test.

## Giới hạn và tài liệu

Bản dữ liệu là bản người dùng cung cấp; không có bản gốc gated tải độc lập để đối chiếu byte. Không biết dữ liệu tiền huấn luyện của API, không thể loại trừ nhiễm benchmark. Tên model do nhà cung cấp công bố. Chưa huấn luyện BiLSTM/PhoBERT trên ViHSD, chưa có tập chat game độc lập. Các kết quả hiện tại chứng minh pipeline và đánh giá đã chạy thật, chưa chứng minh hệ thống đủ tốt để tự động kiểm duyệt.

- Hướng dẫn nhãn: https://arxiv.org/html/2103.11528v4, Bảng 1.
- Nguồn dataset: https://huggingface.co/datasets/uitnlp/vihsd.
- API: `remote_vihsd_sample300_v2/metrics.json`, `confusion_matrix.csv`, `predictions.csv`, `evaluation_manifest.json`.
- Baseline: `baseline/test/metrics.json`, `baseline/dev_results.csv`; audit ở `data/processed/audit.json`.
