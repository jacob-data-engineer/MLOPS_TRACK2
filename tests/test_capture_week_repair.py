import copy

import pytest

from ppe_pipeline_api.data import capture_week_repair as repair, sql_explorer as sx


def test_dates_split_pairs_and_missing_dates_are_reported():
    images = [dict(uri='s3://s/'+name+'.jpg', key=name+'.jpg', captured_at=date)
              for name, date in [('a','2026-03-18T00:00:00+00:00'),
                                 ('b','2026-03-28T00:00:00+00:00'), ('c','')]]
    rows, undated = repair.plan(images, source_prefix='source', root='weeks',
        config={'week_cutoff':'2026-03-21T00:00:00Z','train_percent':100},
        existing=['weeks/week_6/images/train/a.jpg','weeks/week_6/labels/train/a.txt'])
    assert [r['week']['name'] for r in rows] == ['week_1','week_3']
    assert rows[0]['old'] == [['weeks/week_6/images/train/a.jpg'], ['weeks/week_6/labels/train/a.txt']]
    assert rows[0]['target'] == ['weeks/week_1/images/train/a.jpg','weeks/week_1/labels/train/a.txt']
    assert undated == ['s3://s/c.jpg']


class S3:
    def __init__(self, fail_label=False):
        self.objects = {k: dict(ETag='"'+str(i)+'"', ContentLength=10,
                               LastModified='before', Metadata={})
                        for i,k in enumerate(['source/a.jpg','source/a.txt','weeks/week_6/a.jpg','weeks/week_6/a.txt'])}
        self.deleted=[]
        self.fail_label=fail_label

    def head_object(self, Bucket, Key):
        return copy.deepcopy(self.objects[Key])

    def copy_object(self, Bucket, Key, CopySource, **kw):
        if self.fail_label and Key=='weeks/week_1/a.txt':
            raise RuntimeError('copy failed')
        self.objects[Key]=copy.deepcopy(self.objects[CopySource['Key']])
        if 'Metadata' in kw:
            self.objects[Key]['Metadata']=kw['Metadata']

    def delete_object(self, Bucket, Key, **kw):
        self.deleted.append(Key)
        del self.objects[Key]


@pytest.mark.parametrize('fail', [False, True])
def test_old_pairs_are_removed_only_after_both_destinations_and_backups_verify(monkeypatch, fail):
    s3=S3(fail)
    monkeypatch.setattr(sx.storage,'s3',s3)
    row=dict(source=['source/a.jpg','source/a.txt'],target=['weeks/week_1/a.jpg','weeks/week_1/a.txt'],
             old=[['weeks/week_6/a.jpg'],['weeks/week_6/a.txt']],captured_at='2026-03-18T00:00:00+00:00',week={'name':'week_1'})
    if fail:
        with pytest.raises(RuntimeError): repair.relocate_pair(s3,'b',row,'backup/run')
        assert not s3.deleted
    else:
        result=repair.relocate_pair(s3,'b',row,'backup/run')
        assert result['verified']==2 and result['removed']==2
        assert all(s3.objects[k]['Metadata']['captured_at']==row['captured_at'] for k in row['target'])
        assert all('backup/run/'+k in s3.objects for keys in row['old'] for k in keys)
    assert all(k in s3.objects for k in row['source'])
