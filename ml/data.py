"""CSV validation and a shared curated split for all selected preprocessing modes."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .preprocessing import LABEL_NAMES, normalize, preprocess


def read_csv(path, text_col='text', label_col='label'):
    mapping = {str(i): i for i in range(3)} | {name: i for i, name in enumerate(LABEL_NAMES)}
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {text_col, label_col}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing {text_col!r} or {label_col!r}')
        for n, row in enumerate(reader, 2):
            label = (row.get(label_col) or '').strip().upper()
            if label not in mapping:
                raise ValueError(f'{path}:{n}: unsupported label {label!r}')
            rows.append({'text': row.get(text_col) or '', 'label': mapping[label]})
    return rows


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(rows)


def curate(splits, modes=('basic',)):
    """Preserve held-out priority test > dev > train; exclude conflicts per split.

    All modes filter the SAME records, enabling fair ablation. Cross-split labels
    never determine retention. This changes the official benchmark; label curated.
    """
    splits = {name: list(rows) for name, rows in splits.items()}
    audit = {'original': {n: len(r) for n, r in splits.items()}, 'steps': []}
    for mode in dict.fromkeys(('basic', *modes)):
        seen = set()
        step = {'mode': mode}
        for name in ('test', 'dev', 'train'):
            rows = splits[name]
            grouped = defaultdict(list)
            empty = 0
            for row in rows:
                key = preprocess(row['text'], mode)
                if not key:
                    empty += 1
                else:
                    grouped[key].append(row)
            clean, conflicts, duplicates, overlap = [], 0, 0, 0
            for key, group in grouped.items():
                if key in seen:
                    overlap += len(group)
                elif len({r['label'] for r in group}) > 1:
                    conflicts += len(group)
                else:
                    clean.append(group[0])
                    duplicates += len(group) - 1
            # Reserve keys even if held-out conflicting rows were quarantined.
            seen.update(grouped)
            splits[name] = clean
            step[name] = {'empty': empty, 'conflict_rows': conflicts,
                          'duplicate_rows': duplicates, 'overlap_rows': overlap}
        audit['steps'].append(step)
    for name, rows in splits.items():
        counts = Counter(row['label'] for row in rows)
        if set(counts) != {0, 1, 2}:
            raise ValueError(f'{name}: all 3 labels must remain after cleaning; got {dict(counts)}')
    audit['final'] = {n: {'rows': len(r), 'labels': dict(Counter(x['label'] for x in r))}
                      for n, r in splits.items()}
    return splits, audit


def save_prepared(output, splits, audit):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_csv(output / f'{name}.csv', rows)
    write_json(output / 'audit.json', audit)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
