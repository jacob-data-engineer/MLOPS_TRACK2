"""Training progress from completed batches, separate from dataset downloads."""
import time


class TrainingProgressReporter:
    def __init__(self, emit=print, clock=time.monotonic, interval=10):
        self.emit, self.clock, self.interval = emit, clock, interval
        self.processed = self.completed = self.batch_images = self.batches = 0
        self.last_report = float('-inf')
        self.enabled = False

    def start(self, trainer):
        # The deployed pipeline has one GPU. Do not publish one rank's count
        # as a global count if a future configuration uses distributed training.
        self.enabled = getattr(trainer, 'world_size', 1) in (0, 1)
        if not self.enabled:
            return
        self.original_preprocess = trainer.preprocess_batch
        def capture(batch):
            result = self.original_preprocess(batch)
            self.batch_images = int(result['img'].shape[0])
            return result
        trainer.preprocess_batch = capture
        self.epoch_start(trainer)
        self.completed = int(getattr(trainer, 'start_epoch', 0))
        self.processed = self.completed * self.per_epoch
        self.report(trainer, force=True)

    def epoch_start(self, trainer):
        if not self.enabled:
            return
        loader = trainer.train_loader
        self.per_epoch = (len(loader) * loader.batch_size if getattr(loader, 'drop_last', False)
                          else len(loader.dataset))
        self.batches = self.batch_images = 0

    def batch_end(self, trainer):
        if not self.enabled:
            return
        self.processed += self.batch_images
        self.batch_images = 0
        self.batches += 1
        self.report(trainer)

    def epoch_end(self, trainer):
        if not self.enabled:
            return
        if self.batches >= len(trainer.train_loader):
            self.completed = int(trainer.epoch) + 1
        self.report(trainer, force=True)

    def end(self, trainer):
        if self.enabled:
            trainer.preprocess_batch = self.original_preprocess
            self.report(trainer, force=True)

    def report(self, trainer, force=False):
        now = self.clock()
        if force or now - self.last_report >= self.interval:
            self.emit(f'TRAIN_PROGRESS images={self.processed} per_epoch={self.per_epoch} '
                      f'epochs_completed={self.completed}', flush=True)
            self.last_report = now


def attach_training_progress(model):
    reporter = TrainingProgressReporter()
    for event, callback in [('on_train_start', reporter.start),
                            ('on_train_epoch_start', reporter.epoch_start),
                            ('on_train_batch_end', reporter.batch_end),
                            ('on_train_epoch_end', reporter.epoch_end),
                            ('on_train_end', reporter.end)]:
        model.add_callback(event, callback)
    return reporter
