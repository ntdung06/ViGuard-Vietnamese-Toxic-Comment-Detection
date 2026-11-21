import json
from pathlib import Path
from .data import file_hash
from .preprocessing import LABEL_NAMES, preprocess


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / 'metadata.json'
        if not path.is_file():
            raise ModelUnavailable('Chưa có mô hình. Hãy huấn luyện và cấu hình TOXIC_MODEL_DIR.')
        try:
            self.metadata = json.loads(path.read_text(encoding='utf-8'))
            if self.metadata['labels'] != LABEL_NAMES or self.metadata['schema_version'] != 1:
                raise ValueError('Unsupported model schema or labels')
            self.kind = self.metadata['kind']
            self.mode = self.metadata['preprocessing']
            if self.kind == 'baseline':
                import joblib
                path = self.directory / 'model.joblib'
                if file_hash(path) != self.metadata['artifact_sha256']:
                    raise ValueError('Model checksum mismatch')
                # Only load trusted locally trained artifacts. joblib is executable serialization.
                self.model = joblib.load(path)
                if list(self.model.classes_) != [0, 1, 2]:
                    raise ValueError('Unexpected model class order')
            elif self.kind == 'bilstm':
                import torch
                from .neural import BiLSTM
                checkpoint = torch.load(self.directory / 'model.pt', map_location='cpu', weights_only=True)
                self.vocab = checkpoint['vocab']
                self.model = BiLSTM(**checkpoint['architecture'])
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model.eval()
                self.max_length = checkpoint['max_length']
            elif self.kind == 'phobert':
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.directory, local_files_only=True)
                self.model.eval()
                self.max_length = self.metadata['max_length']
            else:
                raise ValueError('Unknown model type')
        except Exception as exc:
            raise ModelUnavailable('Không thể nạp mô hình. Kiểm tra artifact và thư viện ML.') from exc

    def predict(self, texts):
        processed = [preprocess(t, self.mode) for t in texts]
        if self.kind == 'baseline':
            probabilities = self.model.predict_proba(processed).tolist()
        else:
            import torch
            with torch.inference_mode():
                if self.kind == 'bilstm':
                    from .neural import encode_batch
                    tokens, lengths = encode_batch(processed, self.vocab, self.max_length)
                    logits = self.model(tokens, lengths)
                else:
                    inputs = self.tokenizer(processed, padding=True, truncation=True,
                                            max_length=self.max_length, return_tensors='pt')
                    logits = self.model(**inputs).logits
                probabilities = logits.softmax(dim=-1).cpu().tolist()
        results = []
        for row in probabilities:
            index = max(range(3), key=row.__getitem__)
            results.append({'label': LABEL_NAMES[index], 'confidence': float(row[index]),
                            'probabilities': dict(zip(LABEL_NAMES, map(float, row))),
                            'model_version': self.metadata['version']})
        return results
