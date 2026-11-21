import argparse
from .data import curate, file_hash, read_csv, save_prepared


def main():
    p = argparse.ArgumentParser(description='Create a shared curated ViHSD split without leakage.')
    for name in ('train', 'dev', 'test'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--text-col', default='text')
    p.add_argument('--label-col', default='label')
    p.add_argument('--modes', nargs='+', default=['basic', 'teencode', 'segmented', 'enhanced'])
    p.add_argument('--out', default='data/processed')
    a = p.parse_args()
    paths = {n: getattr(a, n) for n in ('train', 'dev', 'test')}
    splits, audit = curate({n: read_csv(path, a.text_col, a.label_col) for n, path in paths.items()}, a.modes)
    audit['source_sha256'] = {n: file_hash(path) for n, path in paths.items()}
    audit['modes'] = a.modes
    audit['evaluation_protocol'] = 'curated split, test > dev > train priority'
    save_prepared(a.out, splits, audit)
    print(audit['final'])


if __name__ == '__main__':
    main()
