# ViGuard

Ứng dụng Django phát hiện bình luận độc hại tiếng Việt với ba nhãn **CLEAN**, **OFFENSIVE**, **HATE**. Repo bao gồm web demo, JSON API, dữ liệu dự đoán tùy chọn qua Django Admin và pipeline nghiên cứu từ dữ liệu CSV đến baseline, BiLSTM, PhoBERT.

## Trạng thái bàn giao

**Đã bổ sung chế độ phân tích qua CheapKeyAI.** Bạn có thể dùng API ngay khi có key hợp lệ thay vì chờ huấn luyện local. Đây là kết quả LLM qua dịch vụ bên thứ ba, không thay thế thí nghiệm BiLSTM/PhoBERT và điểm đánh giá ViHSD của đồ án.

- Đã triển khai mã nguồn ứng dụng và quy trình huấn luyện, đánh giá, xuất/nạp model.
- Đã tải dữ liệu từ bucket người dùng cung cấp `hf://buckets/KHoa121/vihsd-bucket`: 24.048 train, 2.672 dev, 6.680 test. Đã huấn luyện baseline thật; artifact nằm trong `artifacts/baseline` trên máy, được đóng gói riêng trong `baseline-vihsd-model.zip`. Source ZIP không kèm dữ liệu gốc hoặc trọng số.
- Khi chưa có model, web vẫn mở được; API dự đoán trả **503** với thông báo rõ ràng. Không có kết quả ngẫu nhiên hay bộ lọc từ khóa đóng giả mô hình.
- Kiểm thử tích hợp dùng dữ liệu tổng hợp rất nhỏ để kiểm tra quy trình; số đo từ fixture không phải kết quả nghiên cứu.
- Xem `docs/VERIFICATION.md` để biết chính xác những phần đã chạy và giới hạn kiểm chứng.

## Kết quả chạy dữ liệu thật ngày 3 tháng 11 năm 2026

| Thí nghiệm | Tập đánh giá | Macro-F1 | Accuracy |
|---|---|---:|---:|
| API gpt-4o-mini theo hướng dẫn nhãn ViHSD | 300 câu ngẫu nhiên từ 6.680 câu test gốc, seed 42 | 0,6055 | 80,67% |
| TF-IDF Logistic Regression do nhóm huấn luyện | 6.567 câu test sau lọc, curated split | 0,6189 | 82,50% |
| Dummy most frequent | Cùng 6.567 câu test sau lọc | 0,3024 | 83,04% |

Hai model được đánh giá trên hai tập khác nhau; bảng này ghi kết quả riêng, chưa phải so sánh trực tiếp để xếp hạng. Điểm API mới đại diện mẫu 300 câu. Xem `reports/VIHSD_RESULTS.md` để đọc quy trình, lỗi và giới hạn. BiLSTM/PhoBERT chưa được huấn luyện trên toàn bộ ViHSD.

Nhãn hiện theo hướng dẫn ViHSD trong bài báo gốc: OFFENSIVE là ngôn từ xúc phạm không nhắm đối tượng cụ thể; HATE bao gồm công kích trực tiếp cá nhân hoặc nhóm. Vì vậy, câu chửi nhắm một người có thể là HATE, không chỉ thù ghét nhóm căn tính. Prompt prototype trước đó hẹp hơn và đã được sửa theo tài liệu gốc trước khi xem bảng điểm API; lượt prototype dừng vì timeout và không được dùng làm kết quả cuối.

## Cài đặt trên Windows

Khuyến nghị **Python 3.12** cho khả năng tương thích giữa Django, PyTorch và thư viện tiếng Việt. Core web/baseline cũng có thể dùng Python mới hơn; các thư viện NLP có thể chưa cung cấp wheel tương ứng.

Mở PowerShell tại thư mục chứa `manage.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Mở **http://127.0.0.1:8000/**. Không cần Node.js. Frontend dùng Django Templates, CSS, JavaScript thuần, không phụ thuộc CDN.

Linux/macOS: `python3.12 -m venv .venv`, thay đường dẫn Python bằng `.venv/bin/python`.

Các lệnh còn lại dùng `python` để ngắn gọn. Hãy kích hoạt venv hoặc thay bằng đường dẫn Python trong venv.

## Phân tích qua API CheapKeyAI

[Tài liệu nhà cung cấp](https://cheapkeyai.shop/docs) quy định base URL là `https://cheapkeyai.shop/v1`; `/keys` là trang quản lý key. Chỉ backend giữ và dùng key, frontend không nhận khóa. Không đưa key vào Git hoặc gửi trong bình luận.

Sau khi cài core requirements và migrate, cấu hình PowerShell:

```powershell
$env:PREDICTION_BACKEND = "remote"
$env:CHEAPKEYAI_BASE_URL = "https://cheapkeyai.shop/v1"
$env:CHEAPKEYAI_MODEL = "gpt-4o-mini"
$secret = Read-Host "Nhập API key" -AsSecureString
$env:CHEAPKEYAI_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
python manage.py runserver
```

Đổi model bằng biến môi trường nếu key có quyền sử dụng model đó. Lượt phân tích gửi bình luận đến CheapKeyAI và có thể tính phí theo tài khoản nhà cung cấp. Batch chế độ API tối đa 10 câu, mỗi request timeout 45 giây, không tự retry để tránh gọi tính phí lặp.

API trả nhãn và giải thích ngắn, `confidence=null`, `probabilities={}`. Giao diện tự ẩn biểu đồ điểm trong chế độ này; không tạo phần trăm tin cậy từ lời tự đánh giá của LLM. Lỗi khóa, số dư, hạn mức và kết quả sai schema được xử lý mà không trả raw response hoặc key ra frontend. Health chỉ xác nhận cấu hình, không chứng nhận số dư hay model đang phục vụ.

Muốn trở lại model huấn luyện của nhóm: đặt `$env:PREDICTION_BACKEND="local"`, cấu hình `TOXIC_MODEL_DIR`, rồi restart Django. Lịch sử lưu được cả hai loại kết quả, vẫn tuân theo `STORE_PREDICTIONS` và quyền sở hữu tài khoản.

## 1 Chuẩn bị ViHSD

Nguồn: [UIT NLP trên Hugging Face](https://huggingface.co/datasets/uitnlp/vihsd), [repo của tác giả](https://github.com/sonlam1102/vihsd). Đọc điều kiện truy cập và trích dẫn bộ dữ liệu trong báo cáo.

Nếu đã được cấp quyền, đặt `HF_TOKEN` trong môi trường rồi chạy:

```powershell
$env:HF_TOKEN = "token-cua-ban"
python -m ml.download_data
Remove-Item Env:HF_TOKEN
```

Không commit token. Nếu tải tự động không được, lấy bản CSV bạn có quyền truy cập và đặt thành `data/raw/train.csv`, `dev.csv`, `test.csv`.

Định dạng mong đợi:

```csv
text,label
"Cảm ơn bạn đã chia sẻ",0
```

Nhãn mặc định `0=CLEAN`, `1=OFFENSIVE`, `2=HATE`; cũng nhận ba nhãn chữ. **Xác nhận ánh xạ với nguồn trước khi chạy.** Nếu CSV gốc có cột `free_text`, `label_id`, dùng hai tham số dưới đây. Không tự động đổi nhãn không rõ nghĩa.

Chạy bản core, chưa cần thư viện tách từ:

```powershell
python -m ml.prepare --train data/raw/train.csv --dev data/raw/dev.csv --test data/raw/test.csv --text-col free_text --label-col label_id --modes basic teencode --out data/processed
python -m ml.train_baseline --data data/processed --modes basic teencode
```

Nếu cột đã tên `text`, `label`, bỏ `--text-col` và `--label-col`.

Pipeline giữ ưu tiên **test > dev > train**, loại dòng rỗng, câu trùng, nhóm nhãn mâu thuẫn trong từng split; kiểm tra lại giao nhau sau từng phép biến đổi. Tất cả mô hình phải dùng cùng snapshot `data/processed`, không chuẩn bị lại với tập mode khác giữa các lần so sánh. Xem `data/processed/audit.json` để kiểm đếm. Đây là **curated split**, không phải benchmark gốc.

## 2 Huấn luyện baseline và dùng trên web

Baseline tìm `C=0.1/1/10`, trọng số lớp `None/balanced`, TF-IDF n-gram từ `(1,2)`. Chọn cấu hình bằng **dev Macro-F1**, chấm test một lần. TF-IDF chỉ fit train.

Sau huấn luyện, model nằm tại `artifacts/baseline/`, web dùng thư mục này mặc định. Khởi động lại Django sau khi thay model.

```powershell
python manage.py runserver
```

Các file báo cáo:

| File | Nội dung |
|---|---|
| `reports/baseline/dev_results.csv` | Mỗi cấu hình và dev Macro-F1 |
| `reports/baseline/test/metrics.json` | Accuracy, Macro-F1, precision/recall/F1/support từng lớp |
| `reports/baseline/test/confusion_matrix.csv` | Hàng nhãn thật, cột dự đoán |
| `reports/baseline/test/errors.csv` | Mẫu dự đoán sai để phân tích định tính |
| `reports/baseline/dummy_test/metrics.json` | Mốc dự đoán lớp phổ biến nhất |
| `artifacts/baseline/metadata.json` | Nhãn, tiền xử lý, cấu hình, phiên bản và hash dữ liệu |

Không dùng test để chọn tham số sau khi đọc lỗi. Nếu thay phương pháp, chốt cách đánh giá trước lần chấm tiếp theo và công khai mọi thay đổi. Xác suất đầu ra chưa được hiệu chỉnh; không diễn giải thành bảo đảm đúng.

## 3 Chuẩn bị đầy đủ bốn nhánh và huấn luyện deep learning

```powershell
python -m pip install -r requirements-deep.txt
python -m ml.prepare --train data/raw/train.csv --dev data/raw/dev.csv --test data/raw/test.csv --text-col free_text --label-col label_id --out data/processed
python -m ml.train_baseline
python -m ml.train_deep --model bilstm --mode segmented --epochs 10 --weighted-loss --out artifacts/bilstm --reports reports/bilstm
python -m ml.train_deep --model phobert --mode segmented --epochs 3 --batch-size 16 --weighted-loss --out artifacts/phobert --reports reports/phobert
```

Deep learning tự dùng CUDA nếu có; `--device cpu` để ép CPU. Nếu GPU thiếu RAM, giảm `--batch-size 8` hoặc `4`. Có clipping gradient, seed, checkpoint tốt nhất theo dev Macro-F1, early stopping và bảng lịch sử huấn luyện.

Gộp bảng so sánh sau khi cả ba model chạy xong:

```powershell
python -m ml.compare artifacts/baseline artifacts/bilstm artifacts/phobert --out reports/comparison.csv
```

Lệnh từ chối so sánh nếu hash dữ liệu khác nhau.

BiLSTM dùng embedding học từ đầu, vocabulary chỉ tạo từ train, packing theo độ dài để padding không làm lệch trạng thái cuối. PhoBERT mặc định `vinai/phobert-base-v2` từ [VinAI](https://huggingface.co/vinai/phobert-base-v2), tải checkpoint khi huấn luyện lần đầu. Đầu vào phải được tách từ; dự án dùng underthesea cho quy trình gọn. Tác giả PhoBERT khuyến nghị VnCoreNLP để sát tiền huấn luyện hơn; ghi lựa chọn segmenter này trong giới hạn nghiên cứu. Kiểm tra giấy phép checkpoint khi sử dụng/phân phối.

Bốn chế độ tiền xử lý:

| Mode | Chuẩn hóa Unicode và khoảng trắng | Teencode | Tách từ |
|---|---|---|---|
| `basic` | Có | Không | Không |
| `teencode` | Có | Có | Không |
| `segmented` | Có | Không | Có |
| `enhanced` | Có | Có | Có |

PhoBERT chỉ nhận `segmented`/`enhanced` trong script để bảo đảm đầu vào tách từ. Để trả lời RQ2, so hai mode này với cùng siêu tham số và cùng snapshot; baseline/BiLSTM có thể chạy bốn mode. Lưu mỗi lần chạy vào thư mục riêng. RQ1 so BiLSTM và PhoBERT trên cùng tập, dùng Macro-F1 và per-class F1. Nên lặp seed 42/43/44 và báo trung bình/độ lệch chuẩn khi có thời gian, không tuyên bố chênh lệch nhỏ là chắc chắn.

Đổi model phục vụ trên web:

```powershell
$env:TOXIC_MODEL_DIR = "artifacts/phobert"
python manage.py runserver
```

Mỗi worker giữ model trong bộ nhớ; cần restart sau khi thay artifact. Không upload hoặc nạp joblib từ người dùng: đây là dạng serialization chỉ dành cho artifact bạn tin cậy.

## 4 Kiểm tra chat game ngoài miền

Tự thu thập một tập độc lập, xin phép khi cần, bỏ thông tin nhận dạng, gán nhãn theo hướng dẫn ViHSD; không trộn vào train/dev. Hai thành viên gán nhãn độc lập, ghi và giải quyết bất đồng trước khi xem dự đoán.

```powershell
python -m ml.evaluate_external --model-dir artifacts/phobert --csv data/game_chat.csv --reference-data data/processed --out reports/game_chat
```

CSV này dùng cột `text,label`. Script chặn rỗng, trùng và giao với train/dev/test theo tiền xử lý của model. Kết quả ngoài miền phải báo riêng; tập nhỏ không đại diện cho mọi chat game.

## API

| Endpoint | Method | Nội dung |
|---|---|---|
| `/api/health/` | GET | Web đang chạy và có tìm thấy metadata không; không chứng nhận model nạp được |
| `/api/predict/` | POST | JSON `{"text":"Cảm ơn bạn"}` |
| `/api/predict-batch/` | POST | JSON `{"texts":["Xin chào","Cảm ơn"]}`, tối đa 50 câu |
| `/api/history/` | GET | 100 dự đoán gần nhất của chính người đang đăng nhập |
| `/admin/` | GET | Đăng nhập quản trị và xem bản ghi được phép |

Giới hạn 2.000 ký tự/câu, body 128 KiB. POST giữ bảo vệ CSRF. Ví dụ Python client lấy cookie và gửi header:

```python
import requests
s = requests.Session()
s.get('http://127.0.0.1:8000/').raise_for_status()
r = s.post('http://127.0.0.1:8000/api/predict/',
           json={'text': 'Cảm ơn bạn đã chia sẻ'},
           headers={'X-CSRFToken': s.cookies['csrftoken']})
print(r.status_code, r.json())
```

Response thành công có `label`, `confidence`, `probabilities`, `model_version`. Lỗi input trả 400, chưa có model trả 503, chưa đăng nhập lịch sử trả 401. Phương thức không hợp lệ trả 405; lỗi CSRF dùng trang 403 chuẩn Django.

## Lưu lịch sử và tài khoản

Mặc định không lưu bình luận. Muốn lưu dự đoán của người đã đăng nhập:

```powershell
python manage.py createsuperuser
$env:STORE_PREDICTIONS = "1"
python manage.py runserver
```

Đăng nhập `/admin/` rồi dùng web trong cùng trình duyệt. Dự đoán ẩn danh không lưu; `/api/history/` chỉ trả dữ liệu của tài khoản hiện tại. Admin có quyền xem dữ liệu theo quyền Django. Chỉ lưu khi nhóm đã thống nhất mục đích, quyền truy cập và thời hạn lưu.

## Kiểm thử

Đánh giá **API đang dùng trên web** với nhãn thật ViHSD: xem [hướng dẫn đánh giá](docs/EVALUATE_VIHSD.md). Lệnh `python -m ml.evaluate_remote` hỗ trợ chọn mẫu bằng seed cố định, checkpoint tiếp tục, Macro-F1, confusion matrix và báo cáo có hash dữ liệu/prompt. Đã hoàn thành đánh giá 300 câu test với seed 42; xem `reports/remote_vihsd_sample300_v2/REPORT.md`.

```powershell
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py test
```

Bao gồm kiểm tra CSRF, JSON, giới hạn input, batch, quyền sở hữu lịch sử, model thiếu/hỏng, Unicode, rò rỉ sau teencode, huấn luyện và reload baseline. BiLSTM tests tự bỏ qua nếu chưa cài torch. Chạy lại sau khi cài deep requirements để kiểm tra thêm. Dữ liệu test tổng hợp không được dùng để báo điểm dự án.

## Chạy ngoài môi trường phát triển

`runserver` chỉ dùng phát triển. Cấu hình môi trường thật rồi dùng Waitress sau reverse proxy HTTPS:

```powershell
$env:DJANGO_DEBUG = "0"
$env:DJANGO_SECRET_KEY = "thay-bang-secret-ngau-nhien-dai"
$env:DJANGO_ALLOWED_HOSTS = "ten-mien-cua-ban"
python manage.py check --deploy
python manage.py collectstatic --noinput
waitress-serve --listen=127.0.0.1:8000 config.wsgi:application
```

Reverse proxy cần phục vụ `/static/` từ `staticfiles/`, giới hạn request/rate và terminate HTTPS. Cấu hình proxy/HTTPS phụ thuộc máy chủ; chưa có deployment công khai trong bàn giao này. Xem [Django deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/). `.env.example` là ví dụ tên biến; ứng dụng đọc biến môi trường hệ điều hành, không tự đọc file `.env`.

## Cấu trúc

```text
config/                 Cấu hình, routes, WSGI và ASGI
detector/               Views, model lịch sử, admin, template, static và migrations
ml/                     Dữ liệu, tiền xử lý, huấn luyện, đánh giá và inference
tests/                  Kiểm thử tích hợp ML
docs/                   Kiến trúc và checklist outcome
data/                   Dữ liệu cục bộ, không commit dữ liệu thật
artifacts/              Model cục bộ, không commit mặc định
reports/                Metrics, confusion matrix và lịch sử huấn luyện
```

Đây là công cụ hỗ trợ đánh giá nội dung cho đồ án. Bình luận thiếu ngữ cảnh, châm biếm và tiếng lóng có thể bị phân loại sai; cần phân tích lỗi và đánh giá miền dữ liệu trước khi sử dụng thực tế.
