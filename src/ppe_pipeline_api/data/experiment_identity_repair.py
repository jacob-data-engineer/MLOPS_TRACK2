"""Repair historical public experiment identities without renumbering audit IDs.

Run explicitly while the dev worker is idle. Allocated registry numbers are never
removed or recycled; cancelled historical submissions without a number remain
unnumbered. Existing pinned baseline files and evaluation results are preserved.
"""
import contextlib
from ..chain import training_variants as variants
from ..training import comparison
from . import catalog, catalog_view


def repair_experiment_identities(*, apply=False):
    catalog.ensure_schema_once()
    jobs = variants.all_records()
    labels, changes = {}, []
    with contextlib.closing(catalog.connect()) as conn:
        cur = conn.cursor()
        cur.execute('SELECT run_id,run_number FROM run_number_registry')
        allocated = dict(cur.fetchall())
        for job in jobs:
            ids = [v['run_id'] for v in job['variants'] if v.get('run_id')]
            ids += [h['run_id'] for v in job['variants'] for h in v.get('history', []) if h.get('run_id')]
            number = allocated.get('experiment-'+job['id']) or job.get('run_number')
            if not number:
                number = max((allocated[i] for i in ids if i in allocated), default=None)
            arn = job.get('comparison_baseline_arn', job.get('baseline_arn'))
            name = job.get('baseline_label', '')
            if arn and not comparison.is_baseline_group(arn.split('/')[-2]):
                name = comparison.baseline_label(arn)
            if arn and not name:
                if arn not in labels:
                    try:
                        labels[arn] = comparison.baseline_label(arn)
                    except Exception:
                        labels[arn] = ''  # Deleted/unreachable registry: never invent a name.
                name = labels[arn]
            changes.append({'id': job['id'], 'run_number': number, 'children': len(ids), 'baseline_name': name})
            if apply:
                if number:
                    job['run_number'] = number
                    for run_id in set(ids):
                        catalog_view.link_experiment_run(cur, run_id, number)
                if name:
                    job['baseline_label'] = name
                    pinned = job.get('comparison_baseline') if 'comparison_baseline_arn' in job else job.get('baseline')
                    if pinned:
                        pinned['label'] = name
                tab = variants.table()
                tab._lock(cur, tab.key(job))
                tab._save(cur, job)
        if apply:
            conn.commit()
    return changes
