"""Aggregate model metadata, refusing comparisons over different dataset snapshots."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dirs', nargs='+')
    p.add_argument('--out', default='reports/comparison.csv')
    a = p.parse_args()
    records, reference = [], None
    for directory in a.model_dirs:
        meta = json.loads((Path(directory) / 'metadata.json').read_text(encoding='utf-8'))
        if reference is not None and meta['data_sha256'] != reference:
            p.error('Dataset hashes differ. Retrain on the SAME curated snapshot before comparing.')
        reference = meta['data_sha256']
        metrics = meta['test_metrics']
        row = {'model': meta['kind'], 'version': meta['version'], 'mode': meta['preprocessing'],
               'test_macro_f1': metrics['macro_f1'], 'accuracy': metrics['accuracy']}
        for label in ('CLEAN', 'OFFENSIVE', 'HATE'):
            row['f1_' + label] = metrics['per_class'][label]['f1-score']
        records.append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(out)


if __name__ == '__main__':
    main()
