import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from ppe_pipeline_api.training import track2_shared as t

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('progress_reporter',ROOT/'model_training/training-container/training_progress.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_actual_batch_sizes_and_epochs_are_counted_separately():
    events=[]
    reporter=module.TrainingProgressReporter(emit=lambda line,**kw:events.append(line),interval=0)
    class Loader:
        dataset=range(5)
        batch_size=2
        drop_last=False
        def __len__(self):return 3
    trainer=SimpleNamespace(world_size=1,train_loader=Loader(),preprocess_batch=lambda x:x,epoch=0,start_epoch=0)
    reporter.start(trainer)
    for epoch in range(2):
        trainer.epoch=epoch
        reporter.epoch_start(trainer)
        for n in [2,2,1]:
            trainer.preprocess_batch({'img':SimpleNamespace(shape=(n,3,64,64))})
            reporter.batch_end(trainer)
        reporter.epoch_end(trainer)
    reporter.end(trainer)
    assert events[-1]=='TRAIN_PROGRESS images=10 per_epoch=5 epochs_completed=2'
    # A stopped partial epoch must not be reported as a completed epoch.
    trainer.epoch=2
    reporter.epoch_start(trainer)
    reporter.batch_images=2
    reporter.batch_end(trainer)
    reporter.epoch_end(trainer)
    assert reporter.completed==2 and reporter.processed==12


def test_download_counts_never_become_training_counts(monkeypatch):
    values={'dataset:images_downloaded':12412,'dataset:total_images':12412}
    monkeypatch.setattr(t,'_get_metric_max',lambda job,name:values.get(name))
    assert t.get_training_images_progress('job',50) is None
    values.update({'train:images_processed':10700,'train:images_per_epoch':10624})
    assert t.get_training_images_progress('job',50)=={'current':10700,'total':531200,'per_epoch':10624}


def test_completed_epochs_are_not_inferred_from_current_epoch(monkeypatch):
    monkeypatch.setattr(t,'sm',SimpleNamespace(describe_training_job=lambda **kw:{'HyperParameters':{'epochs':'50'}}))
    monkeypatch.setattr(t,'_get_metric_max',lambda job,name:{'train:epoch':3,'train:epochs_completed':2}.get(name))
    assert t.get_epoch_progress('job')=={'current':2,'total':50,'completed':True}


def test_metric_definitions_read_reporter_line():
    import re
    doc=json.loads((ROOT/'model_training/custom_training/pipeline_open.json').read_text())
    step=next(s for s in doc['Steps'] if s['Type']=='Training')
    metrics={m['Name']:m['Regex'] for m in step['Arguments']['AlgorithmSpecification']['MetricDefinitions']}
    line='TRAIN_PROGRESS images=12 per_epoch=5 epochs_completed=2'
    for name,value in [('images_processed',12),('images_per_epoch',5),('epochs_completed',2)]:
        assert int(re.search(metrics['train:'+name],line).group(1))==value
