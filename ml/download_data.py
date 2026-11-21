"""Download the public ViHSD CSV files; pin their repository revision in provenance."""
import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from .data import file_hash, write_json


def fetch(url):
    headers = {'User-Agent': 'ViGuard-student-project/1.0'}
    request = Request(url, headers=headers)
    if os.getenv('HF_TOKEN'):
        # Do not forward the bearer token when the Hub redirects to file storage.
        request.add_unredirected_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
    try:
        with urlopen(request, timeout=120) as response:
            content = response.read(20 * 1024 * 1024 + 1)
            if len(content) > 20 * 1024 * 1024:
                raise ValueError('Download exceeds the expected 20 MiB limit')
            return content
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError('Dataset access denied. Obtain authorized access on Hugging Face and set HF_TOKEN, '
                               'or place your authorized CSV copies in data/raw. Do not share your token.') from None
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/raw')
    a = p.parse_args()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    info = json.loads(fetch('https://huggingface.co/api/datasets/uitnlp/vihsd'))
    revision = info['sha']
    provenance = {'dataset': 'uitnlp/vihsd', 'revision': revision,
                  'source': 'https://huggingface.co/datasets/uitnlp/vihsd', 'files': {}}
    for name in ('train', 'dev', 'test'):
        path = root / f'{name}.csv'
        url = f'https://huggingface.co/datasets/uitnlp/vihsd/resolve/{revision}/{name}.csv'
        if path.exists():
            raise FileExistsError(f'{path} already exists. Use a new output directory to avoid overwriting.')
        content = fetch(url)
        if len(content) > 20 * 1024 * 1024:
            raise ValueError('Unexpectedly large CSV')
        path.write_bytes(content)
        provenance['files'][path.name] = {'url': url, 'sha256': file_hash(path)}
    write_json(root / 'provenance.json', provenance)
    print(f'Downloaded ViHSD at revision {revision}; inspect headers before preparation.')


if __name__ == '__main__':
    main()
