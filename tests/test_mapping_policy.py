
from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from feature_extractor.config_validator import validate_config
from feature_extractor.mapping_policy import assess_mapping, controller_evidence
from feature_extractor.checked_execution import execute_frames
from feature_extractor.feature_registry import output_path
from feature_extractor.mapped_execution import diagnostic_path

DATE = '2025-09-07'


def config(tmp_path, content=None, decision=None):
    p = tmp_path/'S.csv'
    p.write_text(content or 'Approach,Lane Type,Phase,Channel,Stopbar Distance\nNorth,Through,2,1,150\nEast,Right,4,,0\n')
    return validate_config(p, signal_id='S', data_date=DATE, decision=decision)


def events(phases=(2,4,12)):
    rows=[]
    for phase in phases:
        for start in (0,20,40):
            shift = 10 if phase == 4 else 0
            for code,delta in ((1,0),(8,4),(10,5),(11,6)):
                rows.append(('S',pd.Timestamp(DATE)+pd.Timedelta(seconds=start+shift+delta),code,phase))
    return pd.DataFrame(rows,columns=['signalID','timeStamp','eventCode','eventParam'])


def evidence(tmp_path, validation):
    p=tmp_path/'reviewed-controller-plan.txt';p.write_text('Synthetic test evidence only')
    return dict(signal_id='S',reviewed_by='test fixture',reviewed_at=DATE,
                source_path=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                config_sha256=validation.config_hash,valid_from=DATE,valid_to=DATE,
                cycle_method='two_barrier_observed_transitions',
                barrier_map={'2':2,'4':1},ring_sequences=[[2,4]])


def test_missing_channel_does_not_discard_identified_phase(tmp_path):
    v=config(tmp_path)
    assert v.status=='invalid' and v.action=='use'
    assert v.capabilities['vehicle_profiles']['valid']
    assert not v.capabilities['front']['valid']
    assert v.capabilities['back']['valid']  # Missing stopbar channel does not corrupt back detectors.
    r=execute_frames('S',DATE,events(),v,tmp_path/'out',['spat','volume','occupancy'])
    assert r['vehicle_phase_profile'].status=='success'
    frame=read_frame(r['vehicle_phase_profile'].output_file_path)
    assert set(frame.phaseNo)=={2,4}
    assert frame.barrierNo.eq(0).all() and not frame.attrs['barrier_mapping_verified']
    assert not frame.attrs['lane_mapping_effective_date_verified']
    assert r['vehicle_cycle_profile'].reason_code=='barrier_mapping_unverified'
    assert r['volume'].reason_code=='barrier_mapping_unverified'
    assert r['occupancy'].reason_code=='missing_channel_values'
    diagnostic=read_frame(diagnostic_path(tmp_path/'out','unmapped_phase','S',DATE,'.parquet'))
    assert set(diagnostic.phaseNo)=={12}
    assert diagnostic.attrs['usage']=='diagnostic_only_not_for_movement_metrics'
    assert not output_path(tmp_path/'out','spat','S',DATE).exists()


def test_phase_above_eight_requires_movement_identity_not_number_guess(tmp_path):
    v=config(tmp_path,'Approach,Lane Type,Phase,Channel,Stopbar Distance\nSW,Through,12,14,0\n')
    assert v.status=='valid' and v.dataframe.phaseNo.iloc[0]==12
    r=execute_frames('S',DATE,events((12,)),v,tmp_path/'out',['spat'])
    assert set(r['vehicle_phase_profile'].data.phaseNo)=={12}
    assert r['vehicle_cycle_profile'].reason_code=='barrier_mapping_unverified'


def test_phase_number_without_movement_is_diagnostic_only(tmp_path):
    v=config(tmp_path,'Phase\n12\n')
    r=execute_frames('S',DATE,events((12,)),v,tmp_path/'out',['spat'])
    assert r['vehicle_phase_profile'].reason_code=='phase_movement_unverified'
    assert not output_path(tmp_path/'out','vehicle_phase_profile','S',DATE).exists()


def test_default_barrier_cannot_become_evidence_by_matching_observations(tmp_path):
    v=config(tmp_path)
    a=assess_mapping(v,'S',DATE,[2,4])
    assert a['cycle_block_reason']=='barrier_mapping_unverified' and a['barrier_map']=={}


def test_unknown_phase_blocks_cycle_even_with_reviewed_plan(tmp_path):
    v=config(tmp_path);v.decision={'controller_mapping':evidence(tmp_path,v)}
    a=assess_mapping(v,'S',DATE,[2,4,12])
    assert a['cycle_block_reason']=='phase_movement_unverified'
    assert not a['barrier_mapping_verified']


@pytest.mark.parametrize('change,reason',[
    ({'signal_id':'other'},'controller_mapping_invalid'),
    ({'valid_to':'2025-09-06'},'controller_mapping_invalid'),
    ({'valid_from':'2025-09-08','valid_to':'2025-09-09'},'controller_mapping_outside_period'),
    ({'source_sha256':'wrong'},'controller_mapping_stale'),
    ({'reviewed_by':''},'controller_mapping_invalid'),
    ({'barrier_map':{'2':2,'4':3}},'controller_mapping_invalid'),
    ({'ring_sequences':[[2,4,12]]},'controller_mapping_invalid'),
    ({'ring_sequences':[[2,4,2]]},'controller_mapping_invalid'),
    ({'config_sha256':'other'},'controller_mapping_invalid'),
])
def test_controller_evidence_requires_identity_dates_content_and_supported_plan(tmp_path,change,reason):
    v=config(tmp_path);v.decision={'controller_mapping':{**evidence(tmp_path,v),**change}}
    assert controller_evidence(v,'S',DATE)[1]==reason


def test_verified_mapping_includes_all_observed_phases(tmp_path):
    v=config(tmp_path);v.decision={'controller_mapping':evidence(tmp_path,v)}
    assert assess_mapping(v,'S',DATE,[2,4])['barrier_mapping_verified']
    v.dataframe.loc[len(v.dataframe)] = ['North','Left',12,3,0]
    a=assess_mapping(v,'S',DATE,[2,4,12])
    assert a['cycle_block_reason']=='phase_barrier_unverified'


def test_valid_evidence_runs_cycle_without_default_map(tmp_path):
    v=config(tmp_path,'Approach,Lane Type,Phase,Channel,Stopbar Distance\nNorth,Through,2,1,150\nEast,Through,4,2,150\n')
    v.decision={'controller_mapping':evidence(tmp_path,v)}
    r=execute_frames('S',DATE,events((2,4)),v,tmp_path/'out',['spat'])
    assert r['vehicle_cycle_profile'].status=='success',r['vehicle_cycle_profile'].to_dict()
    assert r['spat'].status=='success',r['spat'].to_dict()
    cycle=read_frame(r['vehicle_cycle_profile'].output_file_path)
    assert cycle.attrs['barrier_mapping_verified']
    assert cycle.attrs['controller_evidence']['source_sha256']==v.decision['controller_mapping']['source_sha256']


def test_missing_detector_phase_does_not_remove_other_phase_identity(tmp_path):
    v=config(tmp_path,'Approach,Lane Type,Phase,Channel,Stopbar Distance\nNorth,Through,2,1,150\nEast,Right,,2,0\n')
    assert v.action=='use' and v.capabilities['vehicle_profiles']['valid']
    r=execute_frames('S',DATE,events((2,4)),v,tmp_path/'out',['spat'])
    assert set(r['vehicle_phase_profile'].data.phaseNo)=={2}
    assert r['vehicle_cycle_profile'].reason_code=='barrier_mapping_unverified'
