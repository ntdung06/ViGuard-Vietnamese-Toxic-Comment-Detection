# Kiểm chứng bàn giao hiện tại

- Bộ 33 kiểm thử đã qua trên Windows/Python 3.14.6: Django, API, quyền lịch sử, pipeline, baseline, BiLSTM/Transformers fixture, đánh giá song song/checkpoint và kiểm tra báo cáo khớp model/prompt. Sau khi thêm lịch sử đổi cỡ batch, 4 test đánh giá được chạy lại và đều qua.
- Django check và kiểm tra migrations không phát hiện lỗi.
- Dữ liệu thật đã tải từ bucket do người dùng cung cấp: 24.048 train, 2.672 dev, 6.680 test. Có provenance/hash ở data/raw.
- Baseline đã huấn luyện thật, test curated 6.567 câu: Macro-F1 0,6189, accuracy 82,50%.
- API đã chấm đủ mẫu ngẫu nhiên 300 câu test gốc, seed 42: Macro-F1 0,6055, accuracy 80,67%.
- Giao diện chỉ công bố điểm API nếu file báo cáo có model/provider/prompt/source hash khớp cấu hình hiện tại.
- Prompt API được chỉnh theo hướng dẫn gán nhãn ViHSD trước khi đọc bảng điểm API; lượt prototype chưa hoàn tất được giữ riêng.

Đọc reports/VIHSD_RESULTS.md để biết support từng lớp, nhầm lẫn, giới hạn và lịch sử thực nghiệm. BiLSTM/PhoBERT mới được kiểm tra plumbing bằng fixture, chưa được huấn luyện trên ViHSD thật. Chưa đánh giá API trên toàn bộ test, chưa triển khai HTTPS công khai. Thư viện tách từ dưới Python 3.14 chưa được kiểm chứng; baseline thật dùng basic/teencode.
