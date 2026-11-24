"""Standalone SageMaker Processing entrypoint, uploaded with each comparison.

Reads conditional S3 objects from a saved validation manifest. No training and
no endpoint deployment. All models use one evaluator and the same class set.
"""
import hashlib
import json
from pathlib import Path


def download(s3, uri, etag, target):
    bucket, key = uri.removeprefix('s3://').split('/', 1)
    response = s3.get_object(Bucket=bucket, Key=key, IfMatch=etag)
    digest = hashlib.sha256()
    with open(target, 'wb') as out:
        for chunk in response['Body'].iter_chunks(chunk_size=1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def class_mapping(expected, actual):
    expected = {int(k): str(v) for k, v in expected.items()}
    actual = {int(k): str(v) for k, v in actual.items()}
    if (sorted(actual) != list(range(len(actual))) or len(set(actual.values())) != len(actual)
            or set(expected.values()) != set(actual.values())):
        raise ValueError('Model classes must exactly match the validation classes; no silent subset comparison.')
    inverse = {v: k for k, v in actual.items()}
    return {k: inverse[v] for k, v in expected.items()}


def counts(matrix, names):
    """Ultralytics rows are predictions, columns are ground truth."""
    per_class = []
    for index, name in sorted(names.items()):
        tp = int(matrix[index, index])
        fp = int(matrix[index, :].sum()) - tp
        fn = int(matrix[:, index].sum()) - tp
        per_class.append({'class': name, 'tp': tp, 'fp': fp, 'fn': fn,
                          'precision': tp / (tp + fp) if tp + fp else 0,
                          'recall': tp / (tp + fn) if tp + fn else 0})
    return per_class


def evaluate(manifest, root, s3):
    import yaml
    import ultralytics
    from ultralytics import YOLO
    root = Path(root)
    (root / 'images/val').mkdir(parents=True, exist_ok=True)
    (root / 'labels/val').mkdir(parents=True, exist_ok=True)
    labels, hashes = [], []
    for index, member in enumerate(manifest['validation']['files']):
        suffix = Path(member['image']).suffix
        ih = download(s3, member['image'], member['image_etag'], root / f'images/val/{index}{suffix}')
        target = root / f'labels/val/{index}.txt'
        lh = download(s3, member['label'], member['label_etag'], target)
        labels.append(target.read_text())
        hashes.append([ih, lh])
    rows = []
    # Checkpoints can carry their own saved validation overrides. Explicitly
    # replace them so imported weights cannot silently filter classes or use
    # different augmentation/NMS/precision settings from the trained model.
    fixed = dict(half=False, augment=False, rect=True, agnostic_nms=False,
                 single_cls=False, classes=None, dnn=False)
    settings = {**manifest['settings'], **fixed}
    for index, entry in enumerate(manifest['models']):
        weights = root / f'model-{index}.pt'
        weight_hash = download(s3, entry['uri'], entry['etag'], weights)
        model = YOLO(str(weights))
        if model.task != 'detect':
            raise ValueError('Baseline comparison currently supports object detection models only.')
        names = {int(k): str(v) for k, v in model.names.items()}
        mapping = class_mapping(manifest['validation']['names'], names)
        for number, text in enumerate(labels):
            lines = []
            for line in text.splitlines():
                if line.strip():
                    parts = line.split()
                    if len(parts) != 5 or int(parts[0]) not in mapping:
                        raise ValueError('Invalid saved detection label.')
                    lines.append(' '.join([str(mapping[int(parts[0])]), *parts[1:]]))
            (root / f'labels/val/{number}.txt').write_text('\n'.join(lines))
        # Label remapping must never reuse the previous model's YOLO cache.
        for cache in (root / 'labels').rglob('*.cache'):
            cache.unlink()
        yaml_path = root / 'data.yaml'
        yaml_path.write_text(yaml.safe_dump({'path': str(root), 'train': 'images/val',
                                            'val': 'images/val', 'names': names}))
        common = dict(data=str(yaml_path), split='val', imgsz=settings['imgsz'],
                      iou=settings['nms_iou'], max_det=settings['max_det'],
                      batch=1, device='cpu', workers=0, plots=True, save_json=False,
                      **fixed,
                      project=str(root / 'results'), verbose=False)
        result = model.val(**common, conf=settings['ap_conf'], name=f'ap-{index}')
        operating = model.val(**common, conf=settings['confidence'], name=f'operating-{index}')
        per_class = counts(operating.confusion_matrix.matrix, names)
        ap = {int(k): float(v) for k, v in zip(result.box.ap_class_index, result.box.ap)}
        for row in per_class:
            cls = next(k for k, v in names.items() if v == row['class'])
            # Absent classes must be blank, never the aggregate AP fallback.
            row['mAP50-95'] = ap.get(cls)
        tp, fp, fn = (sum(r[k] for r in per_class) for k in ('tp', 'fp', 'fn'))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        rows.append({'id': entry['id'], 'role': entry['role'], 'label': entry['label'],
                     'weights_sha256': weight_hash, 'mAP50-95': float(result.box.map),
                     'mAP50': float(result.box.map50), 'precision': precision, 'recall': recall,
                     'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0,
                     'tp': tp, 'fp': fp, 'fn': fn, 'per_class': per_class})
    baseline = next(r for r in rows if r['role'] == 'baseline')
    for row in rows:
        row['delta_pp'] = {k: 100 * (row[k] - baseline[k]) for k in
                           ('mAP50-95', 'mAP50', 'precision', 'recall', 'f1')}
    return {'rows': rows, 'settings': settings, 'ultralytics_version': ultralytics.__version__,
            'validation_images': len(labels), 'snapshot_id': manifest['validation']['snapshot_id'],
            'validation_sha256': hashlib.sha256(json.dumps(hashes).encode()).hexdigest(),
            'dataset_version_id': manifest['validation']['dataset_version_id']}


def main():
    import boto3
    s3 = boto3.client('s3')
    manifest = json.loads(Path('/opt/ml/processing/input/manifest.json').read_text())
    output = Path('/opt/ml/processing/output')
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = {'status': 'Completed', **evaluate(manifest, '/opt/ml/processing/work', s3)}
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
        unavailable = isinstance(exc, (ValueError, FileNotFoundError)) or code in ('NoSuchKey', '404', 'PreconditionFailed', '412')
        result = {'status': 'Unavailable' if unavailable else 'Failed', 'error': str(exc)[:1500]}
    (output / 'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
