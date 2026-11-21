# Ánh xạ với yêu cầu đồ án

| Yêu cầu | Thành phần triển khai | Bằng chứng cần bổ sung khi chạy dữ liệu thật |
|---|---|---|
| Phát biểu bài toán và ba RQ | Tài liệu tuần 1–3 của nhóm, README | Bản báo cáo cuối kỳ |
| Nạp và làm sạch dữ liệu | `ml.download_data`, `ml.prepare` | Nguồn dữ liệu hợp lệ và `audit.json` |
| Transformation và features | `ml.preprocessing`, TF-IDF trong baseline | Bảng ablation trên cùng snapshot |
| Splitting strategy | `ml.data.curate`, `ml.protocol` | Số mẫu còn lại và kiểm tra giao nhau |
| Lý do chọn mô hình | Nội dung tuần 3, BiLSTM so với PhoBERT | Giới hạn, chi phí và kết quả chạy thật |
| Baseline metrics | `ml.train_baseline`, `ml.evaluation` | `reports/baseline/test/metrics.json` |
| Deep learning | `ml.neural`, `ml.train_deep` | Artifact, lịch sử train/dev, test metrics |
| RQ1 PhoBERT so với BiLSTM | Hai kiểu model và cùng giao thức đánh giá | Bảng Macro-F1, per-class F1, nhiều seed nếu có |
| RQ2 tác động tiền xử lý | basic/teencode/segmented/enhanced | Chạy cấu hình có kiểm soát, không thay nhiều yếu tố cùng lúc |
| RQ3 chat game | `ml.evaluate_external` | Tập chat độc lập đã gán nhãn, nguồn và phân tích lỗi |
| Sản phẩm chạy được | Django Templates và JSON API | Khởi động web với model thật đã huấn luyện |
| GitHub 1 điểm | Cấu trúc repo, README, requirements, tests | Nhóm tự tạo repo và push; chưa push trong phiên này |

Mã nguồn hỗ trợ thực hiện outcome; mã nguồn chưa tự thay thế thí nghiệm, phân tích và báo cáo của nhóm. Trước khi nộp, kiểm tra lại tài liệu tham khảo và nêu rõ mức độ hỗ trợ AI theo quy định của môn học.
