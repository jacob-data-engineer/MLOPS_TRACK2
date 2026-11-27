import pytest
from ppe_pipeline_api.data import central_recipe as cr


def version(count):
    return {'central_recipe': {'bucket': 'b', 'folder': '_training_inputs/'+'a'*32+'/dataset',
                               'store_uri': 's3://b/store'},
            'train_images': count, 'val_images': 0,
            'file_provenance': [{f'central_{kind}': f's3://b/store/objects/replay/{i}.{kind}' for kind in ('image','label')}
                                | {'central_image_etag': 'ok', 'central_label_etag': 'ok'} for i in range(count)]}


def test_missing_first_file_does_not_check_entire_dataset(monkeypatch):
    calls=[]
    def head(bucket,key):
        calls.append(key)
        return None
    monkeypatch.setattr(cr,'_head',head)
    with pytest.raises(ValueError,match='missing or changed'):
        cr.verify_central(version(10000))
    assert len(calls)<=16


def test_success_still_verifies_every_image_and_label(monkeypatch):
    calls=[]
    def head(bucket,key):
        calls.append(key)
        return {'ETag':'ok'}
    monkeypatch.setattr(cr,'_head',head)
    cr.verify_central(version(37))
    assert len(calls)==74 and len(set(calls))==74
