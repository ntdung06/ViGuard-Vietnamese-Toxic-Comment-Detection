import json
import logging
from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST
from ml.inference import ModelUnavailable
from .models import Prediction
from .services import get_predictor, get_verified_evaluation

logger = logging.getLogger(__name__)


def error(message, status=400):
    return JsonResponse({'error': message}, status=status)


def payload(request):
    if request.content_type != 'application/json':
        raise ValueError('Yêu cầu Content-Type: application/json.')
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError, RequestDataTooBig):
        raise ValueError('Nội dung JSON không hợp lệ hoặc quá lớn.')
    if not isinstance(data, dict):
        raise ValueError('Nội dung phải là một JSON object.')
    return data


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Bình luận phải là chuỗi không rỗng.')
    if len(text) > settings.MAX_COMMENT_LENGTH:
        raise ValueError(f'Bình luận tối đa {settings.MAX_COMMENT_LENGTH} ký tự.')
    return text.strip()


def run_predictions(request, texts):
    try:
        results = get_predictor().predict(texts)
    except ModelUnavailable as exc:
        return error(str(exc), 503)
    except Exception:
        logger.exception('Inference failed')
        return error('Không thể dự đoán lúc này. Vui lòng kiểm tra cấu hình mô hình.', 503)
    if settings.STORE_PREDICTIONS and request.user.is_authenticated:
        with transaction.atomic():
            Prediction.objects.bulk_create([
                Prediction(owner=request.user, text=text, **result)
                for text, result in zip(texts, results)
            ])
    return results


@ensure_csrf_cookie
@require_GET
def home(request):
    return render(request, 'detector/home.html', {'remote_backend': settings.PREDICTION_BACKEND == 'remote',
                                                'evaluation': get_verified_evaluation()})


@require_GET
def health(request):
    # Do not load large ML weights during a health probe.
    remote = settings.PREDICTION_BACKEND == 'remote'
    exists = bool(settings.CHEAPKEYAI_API_KEY) if remote else (settings.MODEL_DIR / 'metadata.json').is_file()
    return JsonResponse({'status': 'ok', 'model_configured': exists,
                         'backend': settings.PREDICTION_BACKEND,
                         'model': settings.CHEAPKEYAI_MODEL if remote else 'local',
                         'evaluation': get_verified_evaluation(),
                         'note': 'Chỉ kiểm tra cấu hình; chưa chứng nhận dự đoán thành công.'})


@require_POST
def predict(request):
    try:
        text = validate_text(payload(request).get('text'))
    except ValueError as exc:
        return error(str(exc))
    result = run_predictions(request, [text])
    return result if isinstance(result, JsonResponse) else JsonResponse(result[0])


@require_POST
def predict_batch(request):
    try:
        texts = payload(request).get('texts')
        if not isinstance(texts, list) or not 1 <= len(texts) <= settings.MAX_BATCH_SIZE:
            raise ValueError(f'texts phải có từ 1 đến {settings.MAX_BATCH_SIZE} bình luận.')
        texts = [validate_text(text) for text in texts]
    except ValueError as exc:
        return error(str(exc))
    results = run_predictions(request, texts)
    return results if isinstance(results, JsonResponse) else JsonResponse({'results': results})


@require_GET
def history(request):
    if not request.user.is_authenticated:
        return error('Cần đăng nhập để xem lịch sử.', 401)
    rows = Prediction.objects.filter(owner=request.user).values(
        'id', 'text', 'label', 'confidence', 'probabilities', 'explanation', 'model_version', 'created_at')[:100]
    return JsonResponse({'results': list(rows)})
