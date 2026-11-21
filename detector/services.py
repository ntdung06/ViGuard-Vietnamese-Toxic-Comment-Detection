from functools import lru_cache
from django.conf import settings
from ml.inference import Predictor


@lru_cache(maxsize=1)
def get_predictor():
    # Reuse one model per worker. Restart workers after replacing the model.
    if settings.PREDICTION_BACKEND == 'remote':
        from ml.remote import RemotePredictor
        return RemotePredictor(settings.CHEAPKEYAI_API_KEY, settings.CHEAPKEYAI_BASE_URL,
                               settings.CHEAPKEYAI_MODEL)
    if settings.PREDICTION_BACKEND != 'local':
        from ml.inference import ModelUnavailable
        raise ModelUnavailable('Cấu hình backend dự đoán không hợp lệ.')
    return Predictor(settings.MODEL_DIR)


def get_verified_evaluation():
    """Show measured scores only when model, provider and prompt match the report."""
    if settings.PREDICTION_BACKEND != 'remote':
        return None
    import hashlib
    import json
    from pathlib import Path
    from ml import remote
    try:
        report = json.loads(Path(settings.REMOTE_EVALUATION_REPORT).read_text(encoding='utf-8'))
        matches = (
            report['model'] == settings.CHEAPKEYAI_MODEL
            and report['base_url'] == settings.CHEAPKEYAI_BASE_URL
            and report['dataset_name'] == 'ViHSD'
            and report['split'] == 'test'
            and report['prompt_sha256'] == hashlib.sha256(remote.INSTRUCTIONS.encode('utf-8')).hexdigest()
            and report['classifier_source_sha256'] == hashlib.sha256(Path(remote.__file__).read_bytes()).hexdigest()
        )
        count, total = report['evaluated_rows'], report['available_rows']
        score, accuracy = report['metrics']['macro_f1'], report['metrics']['accuracy']
        if not matches or not 0 < count <= total or not 0 <= score <= 1 or not 0 <= accuracy <= 1:
            return None
        return {'samples': count, 'total': total, 'macro_f1': score, 'accuracy': accuracy,
                'sampling': report['sampling'], 'completed_at_utc': report['completed_at_utc']}
    except (OSError, ValueError, KeyError, TypeError):
        return None
