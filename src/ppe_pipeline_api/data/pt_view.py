"""Bounded, non-executing inspection of ZIP-format PyTorch checkpoints."""
import io
import json
import pickle
import pickletools
import zipfile
from collections import OrderedDict

MAX_FILE = 512 * 1024 * 1024
MAX_PICKLE = 16 * 1024 * 1024


class Inert:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        # Never import checkpoint globals or call their constructors/reducers.
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        return Inert

    def persistent_load(self, pid):
        return Inert()


def plain(value, depth=0):
    if depth > 8:
        return None
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (dict, OrderedDict)):
        return {str(k): plain(v, depth+1) for k, v in list(value.items())[:1000]
                if type(k) in (str, int) and not isinstance(v, Inert)}
    if isinstance(value, (list, tuple)):
        return [plain(v, depth+1) for v in value[:1000] if not isinstance(v, Inert)]
    return None


def inspect(file, filename):
    if not filename.lower().endswith('.pt'):
        raise ValueError('Choose a .pt checkpoint.')
    file.seek(0, 2)
    size = file.tell()
    if not size or size > MAX_FILE:
        raise ValueError('Choose a non-empty checkpoint up to 512 MB.')
    file.seek(0)
    try:
        with zipfile.ZipFile(file) as archive:
            paths = [p for p in archive.infolist() if p.filename.endswith('/data.pkl') or p.filename == 'data.pkl']
            if len(paths) != 1 or paths[0].file_size > MAX_PICKLE:
                raise ValueError('Unsupported checkpoint layout or model metadata exceeds 16 MB.')
            payload = archive.read(paths[0])
            if any(op.name in ('EXT1', 'EXT2', 'EXT4') for op, arg, pos in pickletools.genops(payload)):
                raise ValueError('Pickle extension opcodes are not supported.')
            checkpoint = Reader(io.BytesIO(payload)).load()
            if not isinstance(checkpoint, dict):
                raise ValueError('No checkpoint dictionary found.')
            model = checkpoint.get('ema') or checkpoint.get('model')
            attrs = vars(model) if isinstance(model, Inert) else model if isinstance(model, dict) else {}
            names = attrs.get('names', checkpoint.get('names'))
            if isinstance(names, (list, tuple)):
                names = dict(enumerate(names))
            classes = []
            if isinstance(names, dict):
                classes = sorted([{'id': int(k), 'name': str(v)} for k, v in names.items()
                                  if isinstance(v, str)], key=lambda r: r['id'])
            metadata = {k: plain(checkpoint[k]) for k in ('date','version','epoch','best_fitness','train_args','train_metrics','license','docs') if k in checkpoint}
            for k in ('yaml','args','task','nc'):
                if k in attrs:
                    metadata[k] = plain(attrs[k])
            embedded = [p for p in archive.infolist() if p.filename.endswith('/extra/track2_metadata.json')]
            if len(embedded) == 1 and embedded[0].file_size <= 8*1024*1024:
                metadata['track2_metadata'] = json.loads(archive.read(embedded[0]))
            return {'filename': filename, 'size_bytes': size, 'classes': classes, 'metadata': metadata,
                    'notice': '' if classes else 'Class names are not recorded in this checkpoint. A weights-only state dictionary cannot supply class names.'}
    except (ValueError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f'Could not inspect checkpoint: {exc}') from exc


def inspect_registered(arn):
    """Read actual checkpoint classes, not the registry's target class labels."""
    import tarfile
    from . import pt_view
    from ..training import track2_shared as t2
    package = t2.sm.describe_model_package(ModelPackageName=arn)
    uri = package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    if not uri.startswith('s3://'):
        raise ValueError('This registered model has no supported S3 artifact.')
    bucket, key = uri[5:].split('/', 1)
    obj = t2.s3.get_object(Bucket=bucket, Key=key)
    body = obj['Body']
    try:
        if obj.get('ContentLength', 0) > pt_view.MAX_FILE:
            raise ValueError('Model artifact exceeds the 512 MB inspection limit.')
        content = body.read(pt_view.MAX_FILE + 1)
    finally:
        body.close()
    if len(content) > pt_view.MAX_FILE:
        raise ValueError('Model artifact exceeds the 512 MB inspection limit.')
    if key.lower().endswith('.pt'):
        result = pt_view.inspect(io.BytesIO(content), key.rsplit('/', 1)[-1])
    else:
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:*') as archive:
            members = [m for m in archive if m.isfile() and m.name.lower().endswith('.pt')]
            preferred = [m for m in members if m.name.rsplit('/', 1)[-1] == 'best.pt']
            member = preferred[0] if len(preferred) == 1 else members[0] if len(members) == 1 else None
            if member is None or member.size > pt_view.MAX_FILE:
                raise ValueError('No unique checkpoint found, or checkpoint exceeds 512 MB.')
            with archive.extractfile(member) as stream:
                result = pt_view.inspect(io.BytesIO(stream.read(pt_view.MAX_FILE + 1)), member.name)
    result['registry'] = {'arn': arn, 'artifact': uri, 'status': package.get('ModelApprovalStatus'),
                          'metadata': package.get('CustomerMetadataProperties', {})}
    return result
