# Khung ghi kết quả nghiên cứu

## Dữ liệu và giao thức

Ghi revision ViHSD, ánh xạ nhãn, hash snapshot, số dòng trước/sau từng bước, tỉ lệ lớp và các sai lệch do curated split. Giải thích vì sao giữ nguyên ranh giới split, chỉ fit biểu diễn trên train, chọn cấu hình trên dev và giới hạn quyền sử dụng dữ liệu.

## Kết quả

Điền từ file metrics thật. Không điền số từ dữ liệu kiểm thử tổng hợp.

| Model | Mode | Seed | Dev Macro-F1 | Test Macro-F1 | F1 CLEAN | F1 OFFENSIVE | F1 HATE | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Dummy | Không áp dụng | Không áp dụng | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |
| TF-IDF LR | Cấu hình được chọn | 42 | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |
| BiLSTM | segmented | 42 | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |
| PhoBERT | segmented | 42 | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy | Chờ chạy |

RQ1: so cùng snapshot và điều kiện; ghi số tham số, thời gian, phần cứng. RQ2: so mode với siêu tham số cố định hoặc giao thức tìm kiếm giống nhau; phân biệt hiệu quả của teencode và tách từ. RQ3: báo riêng số mẫu từng lớp và hiệu quả ngoài miền; không đánh đồng với ViHSD test.

## Phân tích lỗi

Đọc confusion matrix, chọn ví dụ điển hình và ẩn danh. Kiểm tra nhầm OFFENSIVE/HATE, phủ định, mỉa mai, từ viết tắt, thiếu ngữ cảnh và bất đồng nhãn. Không chỉnh model để chữa riêng những câu test đã xem.

## Giới hạn và công việc tiếp

Ghi nhãn chủ quan, lệch miền mạng xã hội/game, lớp lệch, giới hạn tài nguyên, segmenter khác tiền huấn luyện PhoBERT và xác suất chưa calibration. Không kết luận triển khai kiểm duyệt tự động chỉ từ một tập test.

## Tài liệu tham khảo

- Luu và cộng sự, 2021, A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts. https://arxiv.org/abs/2103.11528
- Nguyen và Nguyen, 2020, PhoBERT Pre-trained language models for Vietnamese. https://aclanthology.org/2020.findings-emnlp.92/
- Dataset và checkpoint: các liên kết trong README.
