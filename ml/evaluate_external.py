"""Evaluate an already selected frozen model on an independently labeled game-chat set."""
import argparse
from .data import read_csv
from .evaluation import evaluate
from .inference import Predictor
from .preprocessing import LABEL_NAMES
from .preprocessing import preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='reports/game_chat')
    p.add_argument('--reference-data', default='data/processed', help='Exclude overlap with train/dev/test')
    a = p.parse_args()
    rows = read_csv(a.csv)
    if not rows:
        p.error('External evaluation set is empty')
    model = Predictor(a.model_dir)
    from pathlib import Path
    reference = set()
    for split in ('train', 'dev', 'test'):
        reference.update(preprocess(r['text'], model.mode) for r in read_csv(Path(a.reference_data) / f'{split}.csv'))
    seen = set()
    for row in rows:
        key = preprocess(row['text'], model.mode)
        if not key or key in seen or key in reference:
            p.error('External set has empty text, duplicates or overlap with ViHSD. Curate it independently first.')
        seen.add(key)
    outputs = []
    for offset in range(0, len(rows), 32):
        outputs.extend(model.predict([r['text'] for r in rows[offset:offset + 32]]))
    print(evaluate([r['label'] for r in rows], [LABEL_NAMES.index(r['label']) for r in outputs],
                   a.out, [r['text'] for r in rows]))


if __name__ == '__main__':
    main()
