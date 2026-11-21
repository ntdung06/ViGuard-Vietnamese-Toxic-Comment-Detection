import argparse
import csv
import importlib.metadata
from pathlib import Path
from .data import file_hash, read_csv, write_json
from .preprocessing import LABEL_NAMES, preprocess
from .evaluation import evaluate


def main():
    import joblib
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline

    p = argparse.ArgumentParser(description='Select on dev; evaluate chosen baseline once on test.')
    p.add_argument('--data', default='data/processed')
    p.add_argument('--out', default='artifacts/baseline')
    p.add_argument('--reports', default='reports/baseline')
    p.add_argument('--modes', nargs='+', default=['basic', 'teencode', 'segmented', 'enhanced'])
    p.add_argument('--min-df', type=int, default=2)
    a = p.parse_args()
    root, out, reports = Path(a.data), Path(a.out), Path(a.reports)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    train, dev, test = [read_csv(root / f'{n}.csv') for n in ('train', 'dev', 'test')]
    # Require an audited pipeline whose filters cover every selected mode.
    from .protocol import validate_prepared
    validate_prepared(root, a.modes)
    y = [r['label'] for r in train]
    dev_y, test_y = [[r['label'] for r in rows] for rows in (dev, test)]
    results, best, best_score, config = [], None, -1, None
    for mode in a.modes:
        x = [preprocess(r['text'], mode) for r in train]
        dev_x = [preprocess(r['text'], mode) for r in dev]
        for c in (0.1, 1.0, 10.0):
            for weight in (None, 'balanced'):
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=a.min_df, max_features=100000)),
                    ('classifier', LogisticRegression(C=c, class_weight=weight, max_iter=2000, random_state=42)),
                ])
                model.fit(x, y)
                score = float(f1_score(dev_y, model.predict(dev_x), labels=[0, 1, 2], average='macro', zero_division=0))
                row = {'mode': mode, 'C': c, 'class_weight': weight, 'dev_macro_f1': score}
                results.append(row)
                if score > best_score:
                    best, best_score, config = model, score, row
    with open(reports / 'dev_results.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    predictions = best.predict([preprocess(r['text'], config['mode']) for r in test])
    metrics = evaluate(test_y, predictions, reports / 'test', [r['text'] for r in test])
    dummy = DummyClassifier(strategy='most_frequent').fit([[0]] * len(y), y)
    evaluate(dev_y, dummy.predict([[0]] * len(dev)), reports / 'dummy_dev')
    evaluate(test_y, dummy.predict([[0]] * len(test)), reports / 'dummy_test')
    artifact = out / 'model.joblib'
    joblib.dump(best, artifact)
    write_json(out / 'metadata.json', {
        'schema_version': 1, 'kind': 'baseline', 'version': 'baseline-' + file_hash(artifact)[:12],
        'labels': LABEL_NAMES, 'preprocessing': config['mode'], 'config': config,
        'artifact_sha256': file_hash(artifact),
        'data_sha256': {n: file_hash(root / f'{n}.csv') for n in ('train', 'dev', 'test')},
        'test_metrics': metrics, 'evaluation_protocol': 'curated split',
        'versions': {n: importlib.metadata.version(n) for n in ('scikit-learn', 'joblib')},
    })
    print({'selected': config, 'test_macro_f1': metrics['macro_f1']})


if __name__ == '__main__':
    main()
