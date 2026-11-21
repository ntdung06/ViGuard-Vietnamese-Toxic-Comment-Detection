import json
import hashlib
import tempfile
from pathlib import Path
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from django.test import TestCase as DjangoTestCase, override_settings
from django.contrib.auth import get_user_model
from ml.inference import ModelUnavailable
from ml.remote import RemotePredictor
from detector.models import Prediction
from detector.services import get_predictor
from detector.services import get_verified_evaluation


def completion(rows):
    return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'results': rows})}}]}).encode())


class RemoteTests(TestCase):
    def test_missing_key_and_invalid_url(self):
        for key, url in [('', 'https://example.com/v1'), ('fake', 'http://example.com/v1'),
                         ('fake', 'https://user:pass@example.com/v1')]:
            with self.assertRaises(ModelUnavailable):
                RemotePredictor(key, url, 'test')

    def test_valid_result_without_invented_probabilities(self):
        model = RemotePredictor('fake-key', 'https://example.com/v1', 'test')
        with patch.object(model.opener, 'open', return_value=completion([
            {'id': 0, 'label': 'CLEAN', 'explanation': 'Góp ý lịch sự.'}])) as call:
            result = model.predict(['Tôi không đồng ý.'])[0]
        self.assertIsNone(result['confidence'])
        self.assertEqual(result['probabilities'], {})
        self.assertEqual(result['label'], 'CLEAN')
        request = call.call_args.args[0]
        self.assertNotIn('Authorization', request.headers)
        self.assertEqual(request.unredirected_hdrs['Authorization'], 'Bearer fake-key')
        sent = json.loads(request.data)
        self.assertEqual(sent['model'], 'test')

    def test_invalid_count_label_id_and_explanation_rejected(self):
        model = RemotePredictor('fake', 'https://example.com/v1', 'test')
        bad = [[], [{'id': 1, 'label': 'CLEAN', 'explanation': 'x'}],
               [{'id': 0, 'label': 'UNKNOWN', 'explanation': 'x'}],
               [{'id': 0, 'label': 'CLEAN', 'explanation': ''}]]
        for rows in bad:
            with patch.object(model.opener, 'open', return_value=completion(rows)):
                with self.assertRaises(ModelUnavailable):
                    model.predict(['test'])

    def test_upstream_errors_are_sanitized(self):
        model = RemotePredictor('fake', 'https://example.com/v1', 'test')
        for status in [401, 402, 403, 404, 429, 500, 302]:
            with patch.object(model.opener, 'open', side_effect=HTTPError('url', status, 'secret', {}, None)):
                with self.assertRaises(ModelUnavailable) as error:
                    model.predict(['test'])
                self.assertNotIn('secret', str(error.exception))
        with patch.object(model.opener, 'open', side_effect=URLError('secret')):
            with self.assertRaises(ModelUnavailable):
                model.predict(['test'])

    def test_truncated_response_rejected(self):
        model = RemotePredictor('fake', 'https://example.com/v1', 'test')
        with patch.object(model.opener, 'open', return_value=BytesIO(
            b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}')):
            with self.assertRaises(ModelUnavailable):
                model.predict(['test'])


@override_settings(PREDICTION_BACKEND='remote', CHEAPKEYAI_API_KEY='fake',
                   CHEAPKEYAI_BASE_URL='https://example.com/v1', CHEAPKEYAI_MODEL='test', STORE_PREDICTIONS=True)
class RemoteApiTests(DjangoTestCase):
    def setUp(self):
        get_predictor.cache_clear()

    def tearDown(self):
        get_predictor.cache_clear()

    def test_remote_health_does_not_expose_secret(self):
        result = self.client.get('/api/health/').json()
        self.assertEqual(result['backend'], 'remote')
        self.assertTrue(result['model_configured'])
        self.assertNotIn('fake', json.dumps(result))

    def test_report_must_match_current_prompt_model_and_source(self):
        from ml import remote
        report = {
            'model': 'test', 'base_url': 'https://example.com/v1', 'dataset_name': 'ViHSD', 'split': 'test',
            'prompt_sha256': hashlib.sha256(remote.INSTRUCTIONS.encode()).hexdigest(),
            'classifier_source_sha256': hashlib.sha256(Path(remote.__file__).read_bytes()).hexdigest(),
            'evaluated_rows': 300, 'available_rows': 6680,
            'metrics': {'macro_f1': 0.5, 'accuracy': 0.7}, 'sampling': 'fixture', 'completed_at_utc': 'fixture',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with override_settings(REMOTE_EVALUATION_REPORT=path):
                self.assertIsNone(get_verified_evaluation())
                path.write_text(json.dumps(report))
                self.assertEqual(get_verified_evaluation()['samples'], 300)
                path.write_text(json.dumps({**report, 'model': 'different'}))
                self.assertIsNone(get_verified_evaluation())

    def test_remote_api_and_nullable_history(self):
        user = get_user_model().objects.create_user('remote-test')
        self.client.force_login(user)
        with patch.object(get_predictor().opener, 'open', return_value=completion([
            {'id': 0, 'label': 'HATE', 'explanation': 'Kêu gọi loại trừ một nhóm theo căn tính.'}])):
            response = self.client.post('/api/predict/', json.dumps({'text': 'test'}), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['confidence'])
        record = Prediction.objects.get()
        self.assertIsNone(record.confidence)
        self.assertEqual(record.label, 'HATE')
        self.assertTrue(record.explanation)
