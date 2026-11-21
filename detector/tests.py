import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from ml.data import curate
from ml.inference import ModelUnavailable, Predictor
from ml.preprocessing import normalize, preprocess
from .models import Prediction

RESULT = {'label': 'CLEAN', 'confidence': 0.8,
          'probabilities': {'CLEAN': 0.8, 'OFFENSIVE': 0.1, 'HATE': 0.1}, 'model_version': 'test-v1'}


class ApiTests(TestCase):
    def post(self, data, url='/api/predict/'):
        return self.client.post(url, json.dumps(data), content_type='application/json')

    def test_home_and_csrf(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.get('/').status_code, 200)
        self.assertIn('csrftoken', client.cookies)
        self.assertEqual(client.post('/api/predict/', '{}', content_type='application/json').status_code, 403)
        with patch('detector.views.get_predictor') as model:
            model.return_value.predict.return_value = [RESULT]
            response = client.post('/api/predict/', '{"text":"xin chào"}', content_type='application/json',
                                   HTTP_X_CSRFTOKEN=client.cookies['csrftoken'].value)
        self.assertEqual(response.status_code, 200)

    def test_input_validation(self):
        for value in ('', '   ', None, 123, 'x' * 2001, ['x']):
            with self.subTest(value=str(value)[:20]):
                self.assertEqual(self.post({'text': value}).status_code, 400)

    def test_malformed_and_non_object_json(self):
        for data in ('{', '[]', 'null', '2'):
            self.assertEqual(self.client.post('/api/predict/', data, content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post('/api/predict/', {'text': 'x'}).status_code, 400)

    def test_post_only(self):
        self.assertEqual(self.client.get('/api/predict/').status_code, 405)

    def test_model_unavailable(self):
        with patch('detector.views.get_predictor', side_effect=ModelUnavailable('Chưa có mô hình.')):
            self.assertEqual(self.post({'text': 'Xin chào'}).status_code, 503)

    def test_unexpected_error_not_leaked(self):
        with patch('detector.views.get_predictor', side_effect=RuntimeError('secret filesystem path')):
            response = self.post({'text': 'Xin chào'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret filesystem path', response.content.decode())

    def test_batch_atomic_validation(self):
        for texts in ([], ['x'] * 51, ['ok', ''], 'text'):
            with patch('detector.views.get_predictor') as model:
                self.assertEqual(self.post({'texts': texts}, '/api/predict-batch/').status_code, 400)
                model.assert_not_called()

    def test_valid_batch_and_no_default_storage(self):
        with patch('detector.views.get_predictor') as model:
            model.return_value.predict.return_value = [RESULT, RESULT]
            response = self.post({'texts': ['chào', 'cảm ơn']}, '/api/predict-batch/')
        self.assertEqual(len(response.json()['results']), 2)
        self.assertEqual(Prediction.objects.count(), 0)

    @override_settings(STORE_PREDICTIONS=True)
    def test_history_is_scoped_to_owner(self):
        user = get_user_model().objects.create_user(username='one', password='long-test-password')
        other = get_user_model().objects.create_user(username='two', password='long-test-password')
        self.client.force_login(user)
        with patch('detector.views.get_predictor') as model:
            model.return_value.predict.return_value = [RESULT]
            self.assertEqual(self.post({'text': 'Xin chào'}).status_code, 200)
        self.assertEqual(Prediction.objects.get().owner_id, user.id)
        self.assertEqual(len(self.client.get('/api/history/').json()['results']), 1)
        self.client.force_login(other)
        self.assertEqual(self.client.get('/api/history/').json()['results'], [])
        self.client.logout()
        self.assertEqual(self.client.get('/api/history/').status_code, 401)

    @override_settings(STORE_PREDICTIONS=True)
    def test_anonymous_comments_not_stored(self):
        with patch('detector.views.get_predictor') as model:
            model.return_value.predict.return_value = [RESULT]
            self.post({'text': 'chào'})
        self.assertEqual(Prediction.objects.count(), 0)

    def test_health_does_not_load_model(self):
        with patch('detector.views.get_predictor') as model:
            self.assertEqual(self.client.get('/api/health/').status_code, 200)
            model.assert_not_called()


class PipelineTests(TestCase):
    def test_preserves_diacritics_negation_and_word_boundaries(self):
        self.assertEqual(normalize('  KHÔNG\nđồng Ý '), 'không đồng ý')
        self.assertEqual(preprocess('ko đồng ý, koko dc', 'teencode'), 'không đồng ý, koko được')
        self.assertEqual(normalize('a\u0301'), 'á')

    def test_unknown_preprocessing_rejected(self):
        with self.assertRaises(ValueError):
            preprocess('test', 'unknown')

    def test_duplicates_conflicts_and_heldout_priority(self):
        splits = {name: [{'text': f'{name} sample class {i}', 'label': i} for i in range(3)]
                  for name in ('train', 'dev', 'test')}
        splits['train'] += [{'text': 'shared', 'label': 0}, {'text': 'SHARED ', 'label': 0},
                            {'text': 'bad', 'label': 1}, {'text': 'bad', 'label': 2},
                            {'text': ' ', 'label': 0}]
        splits['test'] += [{'text': 'shared', 'label': 2}]
        clean, audit = curate(splits, ['basic'])
        self.assertNotIn('shared', [r['text'] for r in clean['train']])
        self.assertIn('shared', [r['text'] for r in clean['test']])
        self.assertNotIn('bad', [r['text'] for r in clean['train']])
        self.assertEqual(audit['steps'][0]['train']['empty'], 1)

    def test_post_transform_collision_removed(self):
        splits = {name: [{'text': f'{name} sample class {i}', 'label': i} for i in range(3)]
                  for name in ('train', 'dev', 'test')}
        splits['train'].append({'text': 'ko đồng ý', 'label': 0})
        splits['test'].append({'text': 'không đồng ý', 'label': 0})
        clean, _ = curate(splits, ['basic', 'teencode'])
        self.assertNotIn('ko đồng ý', [r['text'] for r in clean['train']])

    def test_missing_model_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelUnavailable):
                Predictor(directory)

    def test_invalid_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'metadata.json').write_text('{}')
            with self.assertRaises(ModelUnavailable):
                Predictor(directory)
