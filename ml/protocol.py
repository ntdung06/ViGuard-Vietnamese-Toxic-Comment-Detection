import json
from pathlib import Path
from .data import read_csv
from .preprocessing import preprocess


def validate_prepared(root, modes):
    root = Path(root)
    audit = json.loads((root / 'audit.json').read_text(encoding='utf-8'))
    if not set(modes).issubset(set(audit['modes']) | {'basic'}):
        raise ValueError('Prepare the shared split with all requested preprocessing modes first.')
    for mode in set(modes) | {'basic'}:
        seen = set()
        for name in ('test', 'dev', 'train'):
            rows = read_csv(root / f'{name}.csv')
            if {r['label'] for r in rows} != {0, 1, 2}:
                raise ValueError(f'{name}: expected all three classes')
            keys = [preprocess(r['text'], mode) for r in rows]
            if '' in keys or len(keys) != len(set(keys)) or seen.intersection(keys):
                raise ValueError(f'{name}: empty, duplicate or cross-split overlap for {mode}')
            seen.update(keys)
