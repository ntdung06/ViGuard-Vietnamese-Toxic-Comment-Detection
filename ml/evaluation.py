from pathlib import Path
import csv
from .data import write_json
from .preprocessing import LABEL_NAMES


def evaluate(labels, predictions, output, texts=None):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        'macro_f1': float(f1_score(labels, predictions, labels=[0, 1, 2], average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(labels, predictions)),
        'per_class': classification_report(labels, predictions, labels=[0, 1, 2],
                                           target_names=LABEL_NAMES, output_dict=True, zero_division=0),
    }
    write_json(output / 'metrics.json', result)
    with open(output / 'confusion_matrix.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true / predicted', *LABEL_NAMES])
        for name, row in zip(LABEL_NAMES, confusion_matrix(labels, predictions, labels=[0, 1, 2])):
            writer.writerow([name, *row])
    if texts is not None:
        with open(output / 'errors.csv', 'w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['text', 'true_label', 'prediction'])
            for text, true, pred in zip(texts, labels, predictions):
                if int(true) != int(pred):
                    writer.writerow([text, LABEL_NAMES[int(true)], LABEL_NAMES[int(pred)]])
    return result
