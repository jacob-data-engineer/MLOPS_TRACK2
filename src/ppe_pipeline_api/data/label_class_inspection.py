"""Read-only, bounded class-ID inspection for source folder previews."""
from concurrent.futures import ThreadPoolExecutor
import math
from .. import pull_control as pc


def inspect_labels(bucket: str, prefix: str, limit: int = 200) -> dict:
    prefix = prefix.strip().strip('/')
    if not bucket or not prefix:
        raise ValueError('Choose a bucket and a labels folder.')
    keys = sorted(o['Key'] for o in pc._list_objects(prefix + '/', bucket)
                  if o['Key'].lower().endswith('.txt'))
    total = len(keys)
    if total > limit:
        keys = [keys[int(i * total / limit)] for i in range(limit)]

    def read(key):
        try:
            text = pc.s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
        except Exception:
            return {}, 0, 1
        counts, invalid = {}, 0
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                values = [float(p) for p in parts]
                ident = values[0]
                if len(values) < 5 or not all(math.isfinite(v) for v in values) or ident < 0 or not ident.is_integer():
                    raise ValueError('Invalid label')
                ident = int(ident)
                counts[ident] = counts.get(ident, 0) + 1
            except (ValueError, OverflowError):
                invalid += 1
        return counts, invalid, 0

    counts, invalid, unreadable = {}, 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for found, bad, failed in pool.map(read, keys):
            invalid += bad
            unreadable += failed
            for ident, count in found.items():
                counts[ident] = counts.get(ident, 0) + count
    return {'counts': {str(k): v for k, v in sorted(counts.items())},
            'total': total, 'scanned': len(keys), 'unreadable': unreadable,
            'invalid_lines': invalid, 'complete': len(keys) == total and not unreadable and not invalid}
