"""Train BiLSTM or PhoBERT, early stop by dev Macro-F1, then evaluate test once."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from .data import file_hash, read_csv, write_json
from .evaluation import evaluate
from .preprocessing import LABEL_NAMES, MODES, preprocess
from .protocol import validate_prepared


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from .neural import BiLSTM, encode_batch, vocabulary

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['bilstm', 'phobert'], required=True)
    p.add_argument('--data', default='data/processed')
    p.add_argument('--out', required=True)
    p.add_argument('--reports', required=True)
    p.add_argument('--mode', choices=MODES, default='segmented')
    p.add_argument('--checkpoint', default='vinai/phobert-base-v2')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--lr', type=float)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--vocab-size', type=int, default=30000)
    p.add_argument('--embedding-dim', type=int, default=128)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--weighted-loss', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.max_length, a.patience) < 1 or a.vocab_size < 3:
        p.error('epochs, batch-size, max-length, patience must be positive; vocab-size >= 3')
    if a.model == 'phobert' and a.mode not in ('segmented', 'enhanced'):
        p.error('PhoBERT requires word-segmented input: use segmented or enhanced.')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    root, out, reports = Path(a.data), Path(a.out), Path(a.reports)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    validate_prepared(root, [a.mode])
    rows = {n: read_csv(root / f'{n}.csv') for n in ('train', 'dev', 'test')}
    datasets = {n: [(preprocess(r['text'], a.mode), r['label']) for r in rs] for n, rs in rows.items()}
    if a.model == 'bilstm':
        vocab = vocabulary([text for text, _ in datasets['train']], a.vocab_size)
        architecture = {'vocab_size': len(vocab), 'embedding_dim': a.embedding_dim,
                        'hidden_dim': a.hidden_dim, 'dropout': 0.3}
        model = BiLSTM(**architecture)
        tokenizer = None
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.checkpoint, num_labels=3,
            id2label=dict(enumerate(LABEL_NAMES)), label2id={n: i for i, n in enumerate(LABEL_NAMES)})
        limit = getattr(model.config, 'max_position_embeddings', 258) - 2
        if a.max_length > limit:
            p.error(f'max-length must be <= {limit} for this checkpoint')
    model.to(device)

    def collate(batch):
        texts, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
        if tokenizer is None:
            tokens, lengths = encode_batch(texts, vocab, a.max_length)
            features = {'tokens': tokens, 'lengths': lengths}
        else:
            features = dict(tokenizer(list(texts), padding=True, truncation=True,
                                      max_length=a.max_length, return_tensors='pt'))
        return features, labels

    loaders = {n: DataLoader(ds, batch_size=a.batch_size, shuffle=(n == 'train'),
                             collate_fn=collate, num_workers=0)
               for n, ds in datasets.items()}
    counts = Counter(label for _, label in datasets['train'])
    weights = torch.tensor([len(datasets['train']) / (3 * counts[i]) for i in range(3)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights if a.weighted_loss else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr or (1e-3 if a.model == 'bilstm' else 2e-5))

    def forward(features):
        features = {k: v.to(device) for k, v in features.items()}
        output = model(**features)
        return output if tokenizer is None else output.logits

    def predict_loader(loader):
        model.eval()
        truth, predictions = [], []
        with torch.inference_mode():
            for features, labels in loader:
                predictions.extend(forward(features).argmax(-1).cpu().tolist())
                truth.extend(labels.tolist())
        return truth, predictions

    best_score, stale, history = -1, 0, []
    checkpoint_path = out / 'best_state.pt'
    for epoch in range(1, a.epochs + 1):
        model.train()
        total, examples = 0.0, 0
        for features, labels in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(features), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
            examples += len(labels)
        truth, predictions = predict_loader(loaders['dev'])
        score = float(f1_score(truth, predictions, labels=[0, 1, 2], average='macro', zero_division=0))
        history.append({'epoch': epoch, 'train_loss': total / examples, 'dev_macro_f1': score})
        print(history[-1], flush=True)
        if score > best_score:
            best_score, stale = score, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_path)
        else:
            stale += 1
            if stale >= a.patience:
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    if a.model == 'bilstm':
        torch.save({'architecture': architecture, 'state_dict': model.cpu().state_dict(),
                    'vocab': vocab, 'max_length': a.max_length}, out / 'model.pt')
        model.to(device)
        artifact_hash = file_hash(out / 'model.pt')
    else:
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        artifact_hash = file_hash(out / 'model.safetensors')
    truth, predictions = predict_loader(loaders['test'])
    metrics = evaluate(truth, predictions, reports / 'test', [r['text'] for r in rows['test']])
    write_json(reports / 'training_history.json', history)
    write_json(out / 'metadata.json', {
        'schema_version': 1, 'kind': a.model, 'version': f'{a.model}-s{a.seed}-{artifact_hash[:12]}',
        'labels': LABEL_NAMES, 'preprocessing': a.mode, 'max_length': a.max_length,
        'best_dev_macro_f1': best_score, 'test_metrics': metrics,
        'evaluation_protocol': 'curated split', 'config': vars(a),
        'data_sha256': {n: file_hash(root / f'{n}.csv') for n in ('train', 'dev', 'test')},
        'versions': {'torch': torch.__version__},
    })
    checkpoint_path.unlink()
    print({'test_macro_f1': metrics['macro_f1'], 'artifact': str(out)})


if __name__ == '__main__':
    main()
