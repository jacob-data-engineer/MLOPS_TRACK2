"""Actual PostgreSQL + S3 integration check in a disposable SQL schema.

Run inside the app's AWS environment. Never submits training or inference.
Only creates/deletes its own generated SQL schema and small S3 test objects.
"""
import concurrent.futures
import io
import json
import os
import time
import uuid
import zipfile


def main():
    ident = uuid.uuid4().hex
    schema = 'track2_dev_check_' + ident
    os.environ['CATALOG_SCHEMA'] = schema
    os.environ['TRACK2_RECORD_BACKEND'] = 'postgres'
    from ppe_pipeline_api import record_store as rs, storage
    from ppe_pipeline_api.data import catalog, catalog_view as cv, catalog_sync as cs
    from ppe_pipeline_api.data.sql_cutover import ensure_record_tables
    bucket = storage.BUCKET
    prefix = '_migration_checks/' + ident
    objects = []
    created = False
    try:
        c = catalog.connect(); q = c.cursor()
        q.execute(f'CREATE SCHEMA {schema}'); c.commit(); c.close(); created = True
        catalog.ensure_schema(report=False)
        c = catalog.connect(); q = c.cursor(); ensure_record_tables(q); c.commit(); c.close()
        source = next(s for s in rs.inventory().values() if s.target == 'pull_control')
        table = rs.Table(source.table)
        def claim(_):
            try:
                table.put_item(Item={'kind':'claim#check','sort':'one'}, ConditionExpression='attribute_not_exists(kind)')
                return True
            except rs.ClientError as exc:
                assert exc.response['Error']['Code'] == 'ConditionalCheckFailedException'
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            assert sum(pool.map(claim, range(16))) == 1
        def increment(_):
            return table.update_item(Key={'kind':'counter','sort':'one'}, UpdateExpression='ADD #v :n',
                                     ExpressionAttributeNames={'#v':'value'}, ExpressionAttributeValues={':n':1},
                                     ReturnValues='UPDATED_NEW')['Attributes']['value']
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            assert sorted(pool.map(increment, range(16))) == list(range(1,17))
        print('atomic claims and counters: passed', flush=True)
        table.put_item(Item={'kind':'expired','sort':'one','expires_at':1})
        assert table.get_item(Key={'kind':'expired','sort':'one'}) == {}
        table.put_item(Item={'kind':'expired','sort':'one','value':'renewed'}, ConditionExpression='attribute_not_exists(kind)')
        assert table.get_item(Key={'kind':'expired','sort':'one'})['Item']['value'] == 'renewed'
        for n in range(7):
            table.put_item(Item={'kind':'page','sort':str(n),'data':[False,None,{'n':n}]})
        seen=[]; kw={'KeyConditionExpression':'#k = :v','ExpressionAttributeNames':{'#k':'kind'},
                     'ExpressionAttributeValues':{':v':'page'},'Limit':3,'ScanIndexForward':False}
        while True:
            page=table.query(**kw);seen += [r['sort'] for r in page['Items']]
            if 'LastEvaluatedKey' not in page:break
            kw['ExclusiveStartKey']=page['LastEvaluatedKey']
        assert seen == list('6543210'), seen
        low=rs.Client();got=low.get_item(TableName=source.table,Key={'kind':{'S':'page'},'sort':{'S':'0'}})
        assert got['Item']['data']['L'][0] == {'BOOL':False}
        print('expiry, pagination, typed values: passed', flush=True)
        for leaf, body in [('images/train/a.jpg',b'actual-image-bytes'),('labels/train/a.txt',b'0 .5 .5 .1 .1')]:
            key=prefix+'/'+leaf; storage.s3.put_object(Bucket=bucket,Key=key,Body=body); objects.append(key)
        cs.members_of=lambda row:[(f's3://{bucket}/{objects[0]}',f's3://{bucket}/{objects[1]}','train','')]
        versions=rs.Table(next(s.table for s in rs.inventory().values() if s.target=='dataset_definitions'))
        versions.put_item(Item={'snapshot_id':'v','version_id':'v','record_type':'dataset_version','names':['helmet'],'state':'ready'})
        audits=rs.Table(next(s.table for s in rs.inventory().values() if s.target=='training_audits'))
        files=rs.Table(next(s.table for s in rs.inventory().values() if s.target=='training_files'))
        # Real audit construction records the file rows BEFORE the audit.
        with files.batch_writer() as writer:
            for n, kind in enumerate(('image','label')):
                writer.put_item(Item={'audit_id':'r','file_key':f'train#{kind}#0','role':'train','kind':kind,
                                     'split':'train','name':'a.jpg' if n==0 else 'a.txt','s3_uri':f's3://{bucket}/{objects[n]}'})
        audits.put_item(Item={'id':'r','record_type':'training_audit','status':'Succeeded','dataset_version_id':'v'})
        assert cv.runs()['runs'][0]['images']==1
        assert cv.run_files('r',kind='')['total']==2
        assert files.query(IndexName='by-file-name',KeyConditionExpression='name = :n',ExpressionAttributeValues={':n':'a.jpg'})['Count']==1
        audits.update_item(Key={'id':'r'},UpdateExpression='SET #s = :s',ExpressionAttributeNames={'#s':'status'},ExpressionAttributeValues={':s':'Stopped'})
        assert cv.runs()['runs'][0]['status']=='Stopped'
        cv._export_bucket=lambda:(bucket,prefix+'/exports')
        export=cv.start_export('run','r')
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            rows=cv._rows('SELECT state,zip_uri,missing FROM exports WHERE id=%s',(export['id'],))
            if rows and rows[0]['state']!='building':break
            time.sleep(.25)
        row=rows[0]; assert row['state']=='ready',row
        assert row['missing']==0
        key=row['zip_uri'].split('/',3)[3]; objects.append(key)
        data=storage.s3.get_object(Bucket=bucket,Key=key)['Body'].read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert z.read('images/train/a.jpg')==b'actual-image-bytes'
            assert z.read('labels/train/a.txt')==b'0 .5 .5 .1 .1'
            assert 'data.yaml' in z.namelist() and 'manifest.csv' in z.namelist()
        files.delete_item(Key={'audit_id':'r','file_key':'train#image#0'})
        assert cv.runs()['runs'][0]['images']==0
        audits.delete_item(Key={'id':'r'})
        assert cv.runs()['total']==0
        print('transactional catalog projection and actual S3 ZIP: passed',flush=True)
        from ppe_pipeline_api.chain.sql_worker import import_outbox
        outbox=f'sql-outbox/{schema}/review_queue/check.json'
        storage.s3.put_object(Bucket=bucket,Key=outbox,Body=json.dumps({'review_id':'check','status':'pending'}).encode())
        objects.append(outbox)
        assert import_outbox()==1
        reviews=rs.Table(next(s.table for s in rs.inventory().values() if s.target=='review_queue'))
        reviews.update_item(Key={'review_id':'check'},UpdateExpression='SET #s = :s',
                            ExpressionAttributeNames={'#s':'status'},ExpressionAttributeValues={':s':'approved'})
        assert import_outbox()==0
        assert reviews.get_item(Key={'review_id':'check'})['Item']['status']=='approved'
        print('worker message import is idempotent and preserves later edits: passed',flush=True)
    finally:
        if objects:
            storage.s3.delete_objects(Bucket=bucket,Delete={'Objects':[{'Key':k} for k in objects]})
        if created:
            # The identifier is generated above, never user input or a production name.
            assert schema == 'track2_dev_check_' + ident and len(ident)==32
            c=catalog.connect();q=c.cursor();q.execute(f'DROP SCHEMA {schema} CASCADE');c.commit();c.close()
    print('SQL integration checks completed; test resources removed.',flush=True)


if __name__=='__main__':
    main()
