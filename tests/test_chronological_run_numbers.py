import sqlite3
from ppe_pipeline_api.data.catalog_view import chronological_runs_sql


def test_run_numbers_follow_dates_across_models_before_filtering():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE training_runs (run_id text, version_id text, started_at text, detector text, execution_arn text, model_package_arn text, detail text, status text, submitted_at text, ended_at text, architecture text, files_synced int, record text, run_number int)')
    db.execute('CREATE TABLE run_number_registry (run_id text PRIMARY KEY, run_number int UNIQUE)')
    db.execute('CREATE TABLE training_run_groups (run_id text PRIMARY KEY, public_run_number int)')
    db.executemany('INSERT INTO run_number_registry VALUES (?,?)', [('first',1),('middle',2),('later',3)])
    db.executemany('INSERT INTO training_runs (run_id, submitted_at, started_at, detector, status, run_number) VALUES (?,?,?,?,?,?)', [
        ('later','2026-11-19','2026-11-19','test','Succeeded',134593),
        ('first','2026-10-01','2026-10-01','helmet','Failed',42),
        ('middle',None,'2026-11-18','other','Stopped',125471),
    ])
    sql=chronological_runs_sql('SELECT run_id,run_number FROM training_runs WHERE detector=?')
    assert db.execute(sql,('test',)).fetchall()==[('later',3)]
    assert db.execute(chronological_runs_sql('SELECT run_number FROM training_runs ORDER BY run_number')).fetchall()==[(1,),(2,),(3,)]
    db.execute("UPDATE training_runs SET run_number=999999 WHERE run_id='later'")
    db.execute("DELETE FROM training_runs WHERE run_id='middle'")
    assert db.execute(chronological_runs_sql('SELECT run_number FROM training_runs ORDER BY run_number')).fetchall()==[(1,),(3,)]
    assert db.execute(sql,('test',)).fetchall()==[('later',3)]


def test_other_catalog_queries_are_unchanged():
    sql='SELECT * FROM dataset_versions WHERE id = %s'
    assert chronological_runs_sql(sql)==sql


def test_variant_jobs_share_public_number_without_consuming_another():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE training_runs (run_id text, version_id text, started_at text, detector text, execution_arn text, model_package_arn text, detail text, status text, submitted_at text, ended_at text, architecture text, files_synced int, record text)')
    db.execute('CREATE TABLE run_number_registry (run_id text PRIMARY KEY, run_number int UNIQUE)')
    db.execute('CREATE TABLE training_run_groups (run_id text PRIMARY KEY, public_run_number int)')
    db.execute("INSERT INTO run_number_registry VALUES ('experiment-one',89)")
    db.executemany('INSERT INTO training_runs(run_id) VALUES (?)', [('variant-a',),('variant-b',)])
    db.executemany('INSERT INTO training_run_groups VALUES (?,89)', [('variant-a',),('variant-b',)])
    assert db.execute(chronological_runs_sql('SELECT run_number FROM training_runs')).fetchall()==[(89,),(89,)]
    assert db.execute('SELECT count(*) FROM run_number_registry').fetchone()[0]==1
