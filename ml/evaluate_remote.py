"""Evaluate a frozen API classifier on an authorized labeled CSV, with resumable batches.

Labels are never sent to the provider. No metrics are published until all selected
rows succeed. Default: entire supplied test set, preserving original duplicates.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from .data import file_hash, read_csv, write_json
from .evaluation import evaluate
from .preprocessing import LABEL_NAMES, normalize
from .remote import INSTRUCTIONS, RemotePredictor


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def plan_rows(rows, sample_size=None, seed=42):
    if not rows or any(not normalize(row['text']) for row in rows):
        raise ValueError('CSV contains no rows or empty comments. Audit it before evaluation.')
    if sample_size is not None and not 1 <= sample_size <= len(rows):
        raise ValueError('sample-size must be between 1 and the number of rows.')
    indices = (sorted(random.Random(seed).sample(range(len(rows)), sample_size))
               if sample_size is not None else list(range(len(rows))))
    if {rows[i]['label'] for i in indices} != {0, 1, 2}:
        raise ValueError('Selected sample must include all three classes. Increase sample-size.')
    return indices


def run_evaluation(rows, indices, predictor, out, protocol, batch_size=10, workers=1, resume_batch_size=None):
    # JSON object keys become strings on disk; canonicalize before resume comparison.
    protocol = json.loads(json.dumps(protocol))
    if not 1 <= batch_size <= 10:
        raise ValueError('batch-size must be between 1 and 10')
    if not 1 <= workers <= 3:
        raise ValueError('workers must be between 1 and 3')
    actual_batch_size = resume_batch_size or batch_size
    if not 1 <= actual_batch_size <= 10:
        raise ValueError('resume-batch-size must be between 1 and 10')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'progress.json'
    completed = {}
    execution_runs = []
    if checkpoint.exists():
        progress = json.loads(checkpoint.read_text(encoding='utf-8'))
        if progress['protocol'] != protocol or progress['indices'] != indices:
            raise ValueError('Checkpoint differs in model, prompt, data or sampling. Use a new output directory.')
        completed = {int(k): v for k, v in progress.get('completed',
                     dict(enumerate(progress.get('predictions', [])))).items()}
        execution_runs = progress.get('execution_runs', [{
            'batch_size': protocol.get('batch_size', batch_size),
            'workers': protocol.get('workers', workers), 'note': 'Imported from existing checkpoint'}])
        if any(k not in range(len(indices)) or type(v) is not int or v not in range(3)
               for k, v in completed.items()):
            raise ValueError('Invalid checkpoint predictions')
    execution_runs.append({'started_at_utc': datetime.now(timezone.utc).isoformat(),
                           'batch_size': actual_batch_size, 'workers': workers,
                           'already_completed': len(completed)})

    def predict_positions(positions):
        results = predictor.predict([rows[indices[i]]['text'] for i in positions])
        if len(results) != len(positions):
            raise ValueError('Incomplete batch')
        return dict(zip(positions, [LABEL_NAMES.index(r['label']) for r in results]))

    def save_progress():
        prefix = []
        for i in range(len(indices)):
            if i not in completed:
                break
            prefix.append(completed[i])
        temporary = checkpoint.with_suffix('.tmp')
        write_json(temporary, {'protocol': protocol, 'indices': indices,
                              'predictions': prefix, 'completed': completed, 'execution_runs': execution_runs})
        temporary.replace(checkpoint)
        print(f'{len(completed)}/{len(indices)} evaluated', flush=True)

    pending_positions = [i for i in range(len(indices)) if i not in completed]
    batches = iter([pending_positions[i:i + actual_batch_size] for i in range(0, len(pending_positions), actual_batch_size)])
    failure = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = set()
        for _ in range(workers):
            batch = next(batches, None)
            if batch is not None:
                running.add(pool.submit(predict_positions, batch))
        while running:
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    completed.update(future.result())
                    save_progress()
                except Exception as exc:
                    failure = failure or exc
            # Drain and checkpoint already-billed in-flight requests on failure.
            if failure is None:
                for _ in done:
                    batch = next(batches, None)
                    if batch is not None:
                        running.add(pool.submit(predict_positions, batch))
    if failure is not None:
        raise failure
    predictions = [completed[i] for i in range(len(indices))]
    truth = [rows[i]['label'] for i in indices]
    metrics = evaluate(truth, predictions, out)
    with (out / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['csv_row_number', 'text_sha256', 'true_label', 'predicted_label'])
        for i, prediction in zip(indices, predictions):
            writer.writerow([i + 2, digest(rows[i]['text']), LABEL_NAMES[rows[i]['label']], LABEL_NAMES[prediction]])
    majority = Counter(r['label'] for r in rows).most_common(1)[0][1] / len(rows)
    manifest = {**protocol, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'execution_runs': execution_runs,
                'evaluated_rows': len(indices), 'available_rows': len(rows),
                'class_counts': {LABEL_NAMES[i]: truth.count(i) for i in range(3)},
                'duplicate_text_rows_in_input': len(rows) - len({normalize(r['text']) for r in rows}),
                'input_majority_class_fraction': majority,
                'metrics': metrics,
                'limitations': [
                    'Provider model identity is self-reported; underlying training data is unknown.',
                    'Possible pretraining contamination cannot be ruled out.',
                    'Prompt taxonomy may differ from ViHSD annotation guidelines.',
                    'A subset evaluation does not establish full-test performance.',
                    'Do not tune prompt/model on these test predictions.',
                ]}
    write_json(out / 'evaluation_manifest.json', manifest)
    lines = [
        '# Kết quả đánh giá mô hình API', '',
        f"Model cấu hình: `{protocol['model']}`; nguồn API: `{protocol['base_url']}`.",
        f"Tập dữ liệu do người chạy khai báo: {protocol['dataset_name']}; split: {protocol['split']}.",
        f'Số mẫu đánh giá: {len(indices)}/{len(rows)}; chọn mẫu: {protocol["sampling"]}.',
        f"SHA256 file dữ liệu: `{protocol['csv_sha256']}`.", '',
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
        f"- Accuracy: {metrics['accuracy']:.4f}", '',
        '| Nhãn | Precision | Recall | F1 | Support |',
        '|---|---:|---:|---:|---:|',
    ]
    for label in LABEL_NAMES:
        item = metrics['per_class'][label]
        lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |")
    lines += ['', 'Xem confusion_matrix.csv để đọc lỗi theo lớp. Báo cáo này đánh giá API bên thứ ba, '
              'không phải mô hình BiLSTM/PhoBERT do nhóm tự huấn luyện. Nếu chỉ chọn một phần dữ liệu, '
              'phải ghi rõ đây là đánh giá trên mẫu. Dữ liệu tiền huấn luyện của API không được biết; '
              'không thể loại trừ khả năng đã gặp dữ liệu benchmark.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--text-col', default='free_text')
    parser.add_argument('--label-col', default='label_id')
    parser.add_argument('--dataset-name', default='ViHSD')
    parser.add_argument('--split', default='test')
    parser.add_argument('--sample-size', type=int, help='Omit for all rows; random sample without replacement otherwise')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--timeout', type=int, default=120, help='Offline request timeout in seconds')
    parser.add_argument('--resume-batch-size', type=int, choices=range(1, 11), help='Retry unfinished rows with smaller batches; recorded in execution history')
    parser.add_argument('--out', default='reports/remote_vihsd')
    parser.add_argument('--plan-only', action='store_true', help='Audit sample, no API calls or billing')
    args = parser.parse_args()
    rows = read_csv(args.csv, args.text_col, args.label_col)
    indices = plan_rows(rows, args.sample_size, args.seed)
    protocol = {
        'dataset_name': args.dataset_name, 'split': args.split,
        'csv_sha256': file_hash(args.csv), 'label_mapping': dict(enumerate(LABEL_NAMES)),
        'model': os.getenv('CHEAPKEYAI_MODEL', 'gpt-4o-mini'),
        'base_url': os.getenv('CHEAPKEYAI_BASE_URL', 'https://cheapkeyai.shop/v1'),
        'prompt_sha256': digest(INSTRUCTIONS),
        'classifier_source_sha256': file_hash(Path(__file__).with_name('remote.py')),
        'sampling': 'full supplied split' if args.sample_size is None else f'random sample, seed={args.seed}',
        'seed': args.seed, 'sample_size': args.sample_size, 'batch_size': args.batch_size,
        'workers': args.workers, 'timeout': args.timeout,
    }
    print(json.dumps({'protocol': protocol, 'selected_rows': len(indices),
                      'class_counts': dict(Counter(rows[i]['label'] for i in indices))}, ensure_ascii=False, indent=2))
    if args.plan_only:
        return
    predictor = RemotePredictor(os.getenv('CHEAPKEYAI_API_KEY', ''), protocol['base_url'], protocol['model'], args.timeout)
    metrics = run_evaluation(rows, indices, predictor, args.out, protocol, args.batch_size, args.workers, args.resume_batch_size)
    print(json.dumps({'macro_f1': metrics['macro_f1'], 'accuracy': metrics['accuracy']}, indent=2))


if __name__ == '__main__':
    main()
