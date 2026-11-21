"""Small synthetic fixtures test plumbing only; never claim these as ViHSD metrics."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch
from django.test import Client, override_settings
from ml.data import curate, save_prepared
from ml.inference import ModelUnavailable, Predictor

ROOT = Path(__file__).resolve().parent.parent


@skipUnless(importlib.util.find_spec('sklearn'), 'Install core ML requirements to run integration test')
class BaselineIntegrationTests(TestCase):
    def test_train_reload_predict_api_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {}
            for name in ('train', 'dev', 'test'):
                splits[name] = [
                    {'text': f'{name} unique{j} {text}', 'label': label}
                    for label, text in enumerate(('cảm ơn chia sẻ hữu ích', 'đồ ngu xúc phạm', 'thù ghét miệt thị'))
                    for j in range(4)
                ]
            splits, audit = curate(splits, ['basic', 'teencode'])
            audit['modes'] = ['basic', 'teencode']
            save_prepared(root / 'data', splits, audit)
            result = subprocess.run([
                sys.executable, '-m', 'ml.train_baseline', '--data', str(root / 'data'),
                '--out', str(root / 'model'), '--reports', str(root / 'reports'),
                '--modes', 'basic', 'teencode', '--min-df', '1',
            ], cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            self.assertEqual(result.returncode, 0, result.stderr)
            predictor = Predictor(root / 'model')
            predictions = predictor.predict(['cảm ơn chia sẻ', 'đồ ngu'])
            self.assertEqual(len(predictions), 2)
            self.assertEqual(set(predictions[0]['probabilities']), {'CLEAN', 'OFFENSIVE', 'HATE'})
            self.assertAlmostEqual(sum(predictions[0]['probabilities'].values()), 1.0)
            self.assertTrue((root / 'reports/test/metrics.json').exists())
            from detector.services import get_predictor
            get_predictor.cache_clear()
            try:
                with override_settings(MODEL_DIR=root / 'model'):
                    response = Client().post('/api/predict/', json.dumps({'text': 'cảm ơn'}), content_type='application/json')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['model_version'], predictions[0]['model_version'])
            finally:
                get_predictor.cache_clear()
            with open(root / 'model/model.joblib', 'ab') as stream:
                stream.write(b'corrupted')
            with self.assertRaises(ModelUnavailable):
                Predictor(root / 'model')


@skipUnless(importlib.util.find_spec('torch'), 'Install deep requirements for BiLSTM test')
class NeuralTests(TestCase):
    def test_packing_padding_and_reload(self):
        import torch
        from ml.neural import BiLSTM, encode_batch, vocabulary
        texts = ['xin chào', '', 'đây là câu dài hơn']
        vocab = vocabulary(texts)
        tokens, lengths = encode_batch(texts, vocab, 10)
        self.assertEqual(lengths.tolist(), [2, 1, 5])
        model = BiLSTM(len(vocab), embedding_dim=8, hidden_dim=4)
        model.eval()
        with torch.inference_mode():
            logits = model(tokens, lengths)
            single_tokens, single_length = encode_batch([texts[0]], vocab, 10)
            single = model(single_tokens, single_length)
        self.assertEqual(tuple(logits.shape), (3, 3))
        self.assertTrue(torch.allclose(logits[0], single[0], atol=1e-6))

    def test_one_epoch_train_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {name: [{'text': f'{name} {j} {text}', 'label': label}
                              for label, text in enumerate(('xin chào', 'xúc phạm', 'thù ghét'))
                              for j in range(2)] for name in ('train', 'dev', 'test')}
            splits, audit = curate(splits, ['basic'])
            audit['modes'] = ['basic']
            save_prepared(root / 'data', splits, audit)
            result = subprocess.run([
                sys.executable, '-m', 'ml.train_deep', '--model', 'bilstm', '--mode', 'basic',
                '--data', str(root / 'data'), '--out', str(root / 'model'),
                '--reports', str(root / 'reports'), '--epochs', '1', '--batch-size', '3',
                '--embedding-dim', '8', '--hidden-dim', '4', '--device', 'cpu',
            ], cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            self.assertEqual(result.returncode, 0, result.stderr)
            result = Predictor(root / 'model').predict(['xin chào'])
            self.assertAlmostEqual(sum(result[0]['probabilities'].values()), 1.0, places=5)


@skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
            'Install transformers and torch to run the local transformer plumbing test')
class TransformerTests(TestCase):
    def test_local_tiny_classifier_train_export_reload(self):
        # Random tiny RoBERTa checks the shared Transformers path, not PhoBERT quality.
        # Stub only segmentation because underthesea may have no wheel for test Python.
        import types
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification
        from ml.preprocessing import LABEL_NAMES
        from ml.train_deep import main
        segmenter = types.SimpleNamespace(word_tokenize=lambda text, format='text': text)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'underthesea': segmenter}):
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            words = ['<pad>', '<unk>', '<s>', '</s>', 'xin', 'chào', 'xúc', 'phạm', 'thù', 'ghét',
                     'train', 'dev', 'test', '0', '1']
            tokenizer_backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token='<unk>'))
            tokenizer_backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, pad_token='<pad>',
                                                unk_token='<unk>', bos_token='<s>', eos_token='</s>')
            tokenizer.save_pretrained(source)
            config = RobertaConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=24, max_position_embeddings=66,
                                   pad_token_id=0, bos_token_id=2, eos_token_id=3, num_labels=3,
                                   id2label=dict(enumerate(LABEL_NAMES)),
                                   label2id={n: i for i, n in enumerate(LABEL_NAMES)})
            RobertaForSequenceClassification(config).save_pretrained(source)
            splits = {name: [{'text': f'{name} {j} {text}', 'label': label}
                             for label, text in enumerate(('xin chào', 'xúc phạm', 'thù ghét'))
                             for j in range(2)] for name in ('train', 'dev', 'test')}
            splits, audit = curate(splits, ['segmented'])
            audit['modes'] = ['segmented']
            save_prepared(root / 'data', splits, audit)
            arguments = ['train_deep', '--model', 'phobert', '--checkpoint', str(source),
                         '--data', str(root / 'data'), '--out', str(root / 'model'),
                         '--reports', str(root / 'reports'), '--epochs', '1', '--batch-size', '3',
                         '--max-length', '32', '--device', 'cpu']
            with patch.object(sys, 'argv', arguments):
                main()
            result = Predictor(root / 'model').predict(['xin chào'])
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(sum(result[0]['probabilities'].values()), 1.0, places=5)
