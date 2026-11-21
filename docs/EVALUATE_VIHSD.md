# Đánh giá model API trên ViHSD

## Trạng thái hiện tại

Đã tải đủ train/dev/test từ bucket do người dùng cung cấp `hf://buckets/KHoa121/vihsd-bucket`. Kết quả API trên 300 câu test cố định: Macro-F1 0,6055, accuracy 80,67%. Xem `reports/remote_vihsd_sample300_v2` và `reports/VIHSD_RESULTS.md`.

Bản bucket được ghi nguồn và hash trong `data/raw/bucket_provenance.json`. Số dòng khớp cấu trúc ViHSD được công bố; bản gốc gated chưa được tải độc lập để đối chiếu từng byte. Dữ liệu gốc không được đưa vào ZIP source.

## Cách lấy dữ liệu cho máy mới

1. Mở https://huggingface.co/datasets/uitnlp/vihsd và đăng nhập hoặc tạo tài khoản Hugging Face.
2. Đọc và chấp nhận điều kiện chia sẻ thông tin/truy cập của nguồn dữ liệu bằng tài khoản của bạn.
3. Tải `test.csv` vào `data/raw/test.csv`. Muốn chạy toàn bộ đồ án, tải thêm `train.csv`, `dev.csv`.
4. Có thể dùng Hugging Face read token qua biến `HF_TOKEN` để chạy `python -m ml.download_data`. Token này khác key CheapKeyAI. Không đưa token vào Git hoặc chat.

Chỉ gửi dữ liệu đến API bên thứ ba khi điều kiện của dữ liệu cho phép việc đó. Đọc hướng dẫn gán nhãn để bảo đảm ánh xạ 0=CLEAN, 1=OFFENSIVE, 2=HATE đúng với bản CSV bạn tải.

## Chốt giao thức trước khi chạy

- Đánh giá **model đang dùng trên web**: `gpt-4o-mini` qua CheapKeyAI, cùng prompt trong `ml.remote`.
- Đầu vào API chỉ có câu bình luận và id, không có nhãn thật.
- Giữ split test gốc để báo kết quả trên split gốc. Không tự loại câu mô hình đoán sai. Câu rỗng cần được kiểm tra và báo chính sách xử lý trước khi tiếp tục; script dừng nếu gặp rỗng.
- Ghi hash CSV, hash prompt, hash mã classifier, model được yêu cầu, endpoint, cỡ mẫu, seed và thời gian hoàn thành.
- Không thay prompt hoặc chọn model dựa trên điểm test. Nếu cần điều chỉnh, dùng dev và công khai các lần đã chấm test.
- Chi phí gọi API phụ thuộc tài khoản nhà cung cấp. Nên kiểm tra kế hoạch bằng `--plan-only`; chọn cỡ mẫu trước khi nhìn kết quả.

## Chạy đánh giá

Tại thư mục project, dùng Python trong venv đã cài `requirements.txt`. Khóa API ở biến môi trường như phần chạy web:

```powershell
$secret = Read-Host "CheapKeyAI key" -AsSecureString
$env:CHEAPKEYAI_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
$env:CHEAPKEYAI_MODEL = "gpt-4o-mini"
$env:CHEAPKEYAI_BASE_URL = "https://cheapkeyai.shop/v1"
```

Kiểm tra kế hoạch 300 mẫu, không gọi API và không tính phí:

```powershell
python -m ml.evaluate_remote --csv data/raw/test.csv --sample-size 300 --seed 42 --plan-only
```

Nếu CSV có cột `text,label`, thêm `--text-col text --label-col label`. Mặc định script nhận `free_text,label_id`.

Chạy mẫu đã chốt:

```powershell
python -m ml.evaluate_remote --csv data/raw/test.csv --sample-size 300 --seed 42 --out reports/remote_vihsd_sample300
```

**Phải ghi “đánh giá trên mẫu 300 câu từ ViHSD test”**, không ghi kết quả trên toàn bộ test. Mẫu được chọn ngẫu nhiên không hoàn lại với seed cố định; nếu không đủ ba lớp script yêu cầu tăng cỡ mẫu. Không chọn lại seed để tăng điểm.

Hoặc đánh giá toàn bộ tập test ngay từ đầu:

```powershell
python -m ml.evaluate_remote --csv data/raw/test.csv --out reports/remote_vihsd_full
```

Chạy mẫu rồi toàn bộ sẽ gọi lại những câu có trong mẫu; đó là hai thí nghiệm riêng và có thể tốn thêm phí. Chọn trước một giao thức phù hợp ngân sách. Không dùng kết quả mẫu để tinh chỉnh rồi gọi toàn bộ là đánh giá độc lập chưa từng xem.

## Đầu ra và khôi phục khi gián đoạn

- `progress.json`: lưu dự đoán sau từng batch, không lưu raw text hoặc key.
- `predictions.csv`: số dòng CSV, hash văn bản, nhãn thật, nhãn dự đoán.
- `metrics.json`: accuracy, Macro-F1 và precision/recall/F1/support từng lớp.
- `confusion_matrix.csv`: hàng nhãn thật, cột nhãn dự đoán.
- `evaluation_manifest.json`: thông tin tái lập và giới hạn nghiên cứu.
- `REPORT.md`: bảng kết quả đưa vào báo cáo sau khi kiểm tra nguồn dữ liệu.

Khi mất kết nối, chạy lại đúng lệnh và output directory để tiếp tục các batch chưa hoàn thành. Không tự retry batch lỗi trong một lần chạy để tránh tính phí lặp. Batch phía nhà cung cấp đã xử lý nhưng mất response vẫn có thể bị tính phí lại khi bạn chủ động chạy tiếp.

Script chỉ tạo bộ metrics khi tất cả mẫu đã chọn có dự đoán hợp lệ. Không loại bỏ âm thầm các câu lỗi để làm điểm đẹp hơn. Đổi model/prompt/dữ liệu/cỡ mẫu phải dùng thư mục đầu ra mới.

## Cách trình bày trung thực

Đây là phép đánh giá LLM API bên thứ ba theo taxonomy ba lớp, không phải mô hình BiLSTM/PhoBERT do nhóm tự huấn luyện. Không biết đầy đủ dữ liệu tiền huấn luyện của API, nên không thể loại trừ nhiễm benchmark. Nhà cung cấp tự công bố tên model; nhóm không xác minh được trọng số nền. Các kết quả này bổ sung cho demo, không thay thế phần thực nghiệm deep learning đã cam kết trong đề tài.

Nguồn chính thức: https://huggingface.co/datasets/uitnlp/vihsd

Bài báo: https://arxiv.org/abs/2103.11528

## Tiếp tục batch lỗi với cỡ nhỏ hơn

Có thể dùng `--workers 3 --timeout 120` cho đánh giá offline. Khi một batch bị timeout, chạy lại đúng protocol và thêm `--resume-batch-size 1` để chỉ xử lý các câu chưa có dự đoán. Lịch sử cỡ batch thực tế được ghi trong `execution_runs`. Phép chạy ngày 3 tháng 11 dùng batch 10 cho 290 câu, batch 1 cho 10 câu cuối sau timeout; không bỏ câu nào, không đổi seed/model/prompt theo kết quả.
