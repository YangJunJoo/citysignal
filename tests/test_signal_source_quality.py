import copy
from types import SimpleNamespace
import pandas as pd
import pytest
from data_downloader.quality import assess_collection
from feature_extractor.checked_runner import quality_sources, settings_for_job, execute_job, job_from_path
from feature_extractor.checked_execution import _result_from_call
from feature_extractor.feature_registry import resolve_features
from feature_extractor.source_quality import scope_source_to_signal
from feature_extractor.split_validation import validate_split_source
from feature_extractor.run_store import RunStore, content_hash
from feature_extractor.result_types import FeatureResult


def source(tmp_path, bad_time=False):
    path = tmp_path/'atspm-2025-9-7.parquet'
    pd.DataFrame({'signalID':['A','A','B',None],
        'timeStamp':['09-07-2025 00:00:00','09-07-2025 00:10:00',
                     'bad' if bad_time else '09-07-2025 00:15:00','09-07-2025 00:20:00'],
        'eventCode':[1,8,1,1],'eventParam':[2,2,2,2]}).to_parquet(path)
    report = assess_collection([path])
    return quality_sources(report)['2025-09-07']


def test_bad_id_does_not_block_a_proven_healthy_signal(tmp_path):
    original=source(tmp_path);snapshot=copy.deepcopy(original)
    scoped=scope_source_to_signal(original,'A')
    assert scoped['status']=='valid' and scoped['source_file_status']=='source_incomplete'
    assert scoped['quality_scope']=='signal' and scoped['excluded_unassigned_rows']==1
    assert not scoped['expected_coverage_verified']
    validate_split_source(pd.DataFrame({'timeStamp':pd.to_datetime(['2025-09-07 00:00','2025-09-07 00:10'])}),scoped)
    assert original==snapshot


def test_bad_timestamp_stays_blocked_for_its_own_signal(tmp_path):
    original=source(tmp_path,bad_time=True)
    assert scope_source_to_signal(original,'A')['status']=='valid'
    assert scope_source_to_signal(original,'B')['status']=='source_incomplete'


@pytest.mark.parametrize('reason',['internal_gap','adjacent_gap','adjacent_overlap','raw_read_failed','missing_raw_columns','outside_expected_interval','implausible_source_date','unknown_reason'])
def test_nonlocal_defects_cannot_be_bypassed(tmp_path,reason):
    original=source(tmp_path);original['reason_codes'].append(reason)
    assert scope_source_to_signal(original,'A')['status']=='source_incomplete'


@pytest.mark.parametrize('signal',['nan','', 'absent'])
def test_unassigned_or_absent_signal_stays_blocked(tmp_path,signal):
    assert scope_source_to_signal(source(tmp_path),signal)['status']=='source_incomplete'


def test_stale_source_and_explicit_rejection_cannot_be_promoted(tmp_path):
    original=source(tmp_path);original['status']='source_quality_stale'
    assert scope_source_to_signal(original,'A')['status']=='source_quality_stale'
    original['status']='source_incomplete';original['reported_status']='valid'
    assert scope_source_to_signal(original,'A')['status']=='source_incomplete'


@pytest.mark.parametrize('change',['hour_count','negative_count','file_conservation','no_signature'])
def test_incomplete_or_corrupt_evidence_stays_blocked(tmp_path,change):
    original=source(tmp_path)
    if change=='hour_count':original['signals']['A']['row_count']+=1
    if change=='negative_count':original['signals']['A']['hourly_counts']={'2025-09-07T00:00:00':-1}
    if change=='file_conservation':original['row_count']+=1
    if change=='no_signature':original.pop('signature')
    assert scope_source_to_signal(original,'A')['status']=='source_incomplete'


def test_recovered_signal_still_enforces_split_identity(tmp_path,monkeypatch):
    original=source(tmp_path)
    raw=tmp_path/'split'/'A'/'atspm-2025-9-7.parquet';raw.parent.mkdir(parents=True)
    # Same signal but wrong count: the quality qualification must not bypass this.
    pd.DataFrame({'signalID':['A'],'timeStamp':[pd.Timestamp('2025-09-07')],'eventCode':[1],'eventParam':[2]}).to_parquet(raw)
    configs=tmp_path/'config';configs.mkdir();cfg=configs/'A.csv'
    cfg.write_text('Phase,Channel,Stopbar Distance,Lane Type\n2,1,0,Through\n')
    job=job_from_path(raw,['spat']);job['config_hash']=content_hash(cfg)
    store=RunStore(tmp_path/'runs.sqlite3');run=store.start_run()
    settings=dict(db_path=store.path,run_id=run,config_dir=str(configs),output_dir=str(tmp_path/'output'),engine='v2',overwrite=True,sources={'2025-09-07':original},decisions={})
    def unexpected(*args,**kwargs):raise AssertionError('Should not extract mismatched split')
    monkeypatch.setattr('feature_extractor.checked_runner.execute_frames',unexpected)
    outcome=execute_job(job,settings_for_job(settings,job))
    assert all(r['reason_code']=='split_source_mismatch' for r in outcome['results'].values())


def test_controller_mapping_mismatch_is_explicitly_blocked(tmp_path):
    def fail():raise ValueError('inconsistent_barrier_mapping: observed green crosses barriers')
    result=_result_from_call('vehicle_cycle_profile',fail,tmp_path/'unused','A')
    assert result.status=='blocked' and result.reason_code=='inconsistent_barrier_mapping'


def test_all_quarantined_cycles_are_blocked_not_successful_empty_output(tmp_path):
    frame=pd.DataFrame();frame.attrs['quarantined_cycle_count']=10
    result=_result_from_call('vehicle_cycle_profile',lambda:frame,tmp_path/'unused','A')
    assert result.status=='blocked' and result.reason_code=='no_valid_cycle_windows'
