import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from ml.evaluate_remote import plan_rows, run_evaluation
from ml.inference import ModelUnavailable


class SamplingTests(TestCase):
    def test_reproducible_random_sample(self):
        rows = [{'text': str(i), 'label': i % 3} for i in range(100)]
        first = plan_rows(rows, 30, 42)
        self.assertEqual(first, plan_rows(rows, 30, 42))
        self.assertEqual(len(first), len(set(first)))

    def test_reject_empty_or_missing_classes(self):
        with self.assertRaises(ValueError):
            plan_rows([])
        with self.assertRaises(ValueError):
            plan_rows([{'text': ' ', 'label': 0}])
        with self.assertRaises(ValueError):
            plan_rows([{'text': 'x', 'label': 0}])


@skipUnless(importlib.util.find_spec('sklearn'), 'Install core ML requirements')
class EvaluationTests(TestCase):
    def test_parallel_batches_keep_original_row_order(self):
        rows = [{'text': str(i), 'label': i % 3} for i in range(18)]
        protocol = {'model': 'fixture', 'base_url': 'https://example.com', 'dataset_name': 'synthetic fixture',
                    'split': 'fixture', 'sampling': 'full fixture', 'csv_sha256': 'fixturehash',
                    'label_mapping': {0: 'CLEAN', 1: 'OFFENSIVE', 2: 'HATE'}}

        class Fake:
            def predict(self, texts):
                return [{'label': ['CLEAN', 'OFFENSIVE', 'HATE'][int(t) % 3]} for t in texts]

        with tempfile.TemporaryDirectory() as directory:
            result = run_evaluation(rows, list(range(18)), Fake(), directory, protocol, batch_size=3, workers=3)
            self.assertEqual(result['accuracy'], 1.0)
            state = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(state['predictions'], [i % 3 for i in range(18)])

    def test_resume_and_report_only_after_completion(self):
        rows = [{'text': str(i), 'label': i % 3} for i in range(6)]
        indices = list(range(6))
        protocol = {'model': 'fixture', 'base_url': 'https://example.com', 'dataset_name': 'synthetic fixture',
                    'split': 'fixture', 'sampling': 'full fixture', 'csv_sha256': 'fixturehash',
                    'label_mapping': {0: 'CLEAN', 1: 'OFFENSIVE', 2: 'HATE'}}

        class Fake:
            fail = True
            seen = []

            def predict(self, texts):
                self.seen.extend(texts)
                if self.fail and texts[0] == '3':
                    raise ModelUnavailable('fixture outage')
                return [{'label': ['CLEAN', 'OFFENSIVE', 'HATE'][int(t) % 3]} for t in texts]

        with tempfile.TemporaryDirectory() as directory:
            model = Fake()
            with self.assertRaises(ModelUnavailable):
                run_evaluation(rows, indices, model, directory, protocol, batch_size=3)
            self.assertFalse((Path(directory) / 'metrics.json').exists())
            self.assertEqual(len(json.loads((Path(directory) / 'progress.json').read_text())['predictions']), 3)
            model.fail = False
            model.seen = []
            result = run_evaluation(rows, indices, model, directory, protocol, batch_size=3, resume_batch_size=1)
            self.assertEqual(model.seen, ['3', '4', '5'])
            self.assertEqual(result['macro_f1'], 1.0)
            state = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(state['execution_runs'][-1]['batch_size'], 1)
            self.assertTrue((Path(directory) / 'REPORT.md').exists())
            with self.assertRaises(ValueError):
                run_evaluation(rows, indices, model, directory, {**protocol, 'model': 'other'}, batch_size=3)
