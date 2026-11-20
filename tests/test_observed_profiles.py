"""State conservation, missing events, barriers and versioned half-open boundaries."""
from types import SimpleNamespace
import pandas as pd
import pytest
from feature_extractor.profiles import (build_vehicle_phase_profile, build_vehicle_cycle_profile,
    validate_cycle_profile, ProfileQualityError, apply_state_quality)
from feature_extractor.data_loader import SignalDataCache
from feature_extractor.extractors.spat import extract_spat

T=pd.Timestamp('2025-09-07')

def events(extra=False):
    rows=[]
    for phase,starts in [(2,[0,20,40,60]),(4,[10,30,50,70]),(8,[10,30,50,70])]:
        for start in starts:
            for code,delta in [(1,0),(8,4),(10,5),(11,6)]:
                rows.append(('S',T+pd.Timedelta(seconds=start+delta),code,phase))
    if extra:
        for code,t in [(1,8),(8,32),(10,33),(11,34)]:rows.append(('S',T+pd.Timedelta(seconds=t),code,10))
    return pd.DataFrame(rows,columns=['signalID','timeStamp','eventCode','eventParam'])

def profiles(e):
    p=build_vehicle_phase_profile(e,'S','2025-09-07')
    return p,build_vehicle_cycle_profile(p,'S','2025-09-07')

def test_tie_order_and_auxiliary_phase_do_not_change_cycle_boundaries():
    _,base=profiles(events())
    p,c=profiles(events(True))
    _,shuffled=profiles(events(True).sample(frac=1,random_state=19))
    pd.testing.assert_frame_equal(c,shuffled)
    pd.testing.assert_series_equal(c.cycleBegin,base.cycleBegin)
    assert ((c.cycleEnd-c.cycleBegin).dt.total_seconds()==20).all()
    # Auxiliary phase spans a boundary and is preserved rather than deleted.
    assert c.greenPhase10.iloc[0]==[(T+pd.Timedelta(seconds=10),T+pd.Timedelta(seconds=30))]
    assert c.greenPhase10.iloc[1]==[(T+pd.Timedelta(seconds=30),T+pd.Timedelta(seconds=32))]
    validate_cycle_profile(c)

def test_missing_yellow_start_retains_clearance_and_marks_unknown():
    e=events();e=e[~((e.eventParam==2)&(e.eventCode==8)&(e.timeStamp==T+pd.Timedelta(seconds=24)))]
    p,c=profiles(e)
    assert c.unknownPhase2.iloc[0]==[(T+pd.Timedelta(seconds=20),T+pd.Timedelta(seconds=25))]
    assert c.redClearancePhase2.iloc[0]==[(T+pd.Timedelta(seconds=25),T+pd.Timedelta(seconds=26))]
    assert all(not(a<T+pd.Timedelta(seconds=25) and b>T+pd.Timedelta(seconds=20)) for a,b in c.redPhase2.iloc[0])
    validate_cycle_profile(c)

def test_incompatible_barrier_map_is_rejected_instead_of_hidden_by_clipping():
    e=events();e.loc[(e.eventParam==2)&(e.eventCode==8)&(e.timeStamp==T+pd.Timedelta(seconds=4)),'timeStamp']=T+pd.Timedelta(seconds=12)
    # Construct a clean known green crossing the other barrier.
    e=e[~((e.eventParam==2)&e.eventCode.isin([10,11])&(e.timeStamp<T+pd.Timedelta(seconds=20)))]
    p=build_vehicle_phase_profile(e,'S','2025-09-07')
    with pytest.raises(ProfileQualityError,match='inconsistent_barrier_mapping'):
        build_vehicle_cycle_profile(p,'S','2025-09-07')

def test_unknown_red_is_not_full_cycle_red_and_traffic_metrics_are_masked():
    e=events();e=e[~((e.eventParam==2)&(e.eventCode==8)&(e.timeStamp==T+pd.Timedelta(seconds=24)))]
    _,c=profiles(e)
    cache=SimpleNamespace(df_vehicle_cycle=c,vehicle_phase_nos=[2,4,8],year=2025,month=9,day=7,save_feature=lambda *a:None)
    s=extract_spat(cache)
    assert s.unknownDurationPhase2.iloc[0]==5
    assert s.stateCoveragePhase2.iloc[0]==.75
    assert s.redDurationPhase2.iloc[0]==14
    m=pd.DataFrame({'cycleNo':[1,2], 'greenVolumePhase2T':[0,1],'greenVolumePhase4T':[3,4]})
    q=apply_state_quality(m,c,'volume')
    assert pd.isna(q.greenVolumePhase2T.iloc[0]) and q.greenVolumePhase4T.iloc[0]==3

def test_boundary_event_has_one_cycle_and_terminal_end_has_none():
    _,c=profiles(events())
    cache=object.__new__(SignalDataCache);cache.df_vehicle_cycle=c;cache._cycle_index_cache={}
    cache.get_merged_events=lambda *a:pd.DataFrame({'timeStamp':[T+pd.Timedelta(seconds=30),T+pd.Timedelta(seconds=70)]})
    out=cache.assign_cycle_index((81,),'back')
    assert out.cycle_idx.tolist()==[1,-1]

def test_red_assignment_remains_on_its_own_sequence():
    p,c=profiles(events())
    q=p[p.phaseNo==2].iloc[0]
    assert q.redBegin==q.redClearanceEnd==T+pd.Timedelta(seconds=6)
    assert q.redEnd==T+pd.Timedelta(seconds=20)


def test_end_yellow_event_prevents_green_from_bridging_a_missing_boundary():
    # Actual source pattern: missing 8, but observed 9 proves green already ended.
    raw=pd.DataFrame({'signalID':['S']*4,'timeStamp':[T+pd.Timedelta(seconds=x) for x in [0,5,20,24]],
                      'eventCode':[1,9,1,8],'eventParam':[2]*4})
    p=build_vehicle_phase_profile(raw,'S','2025-09-07')
    assert pd.isna(p.iloc[0].greenEnd)
    assert p.iloc[1].greenEnd==T+pd.Timedelta(seconds=24)


def test_optional_end_events_preserve_known_color_and_leave_intermediate_gap_unknown():
    raw=pd.DataFrame({'signalID':['S']*9,'timeStamp':[T+pd.Timedelta(seconds=x) for x in [0,4,7,10,11,12,13,19,20]],
                      'eventCode':[1,7,8,9,10,11,12,0,1],'eventParam':[2]*9})
    p=build_vehicle_phase_profile(raw,'S','2025-09-07');r=p.iloc[0]
    assert r.greenEnd==T+pd.Timedelta(seconds=4)
    assert r.yellowBegin==T+pd.Timedelta(seconds=7) and r.yellowEnd==T+pd.Timedelta(seconds=10)
    assert r.redClearanceBegin==T+pd.Timedelta(seconds=11) and r.redClearanceEnd==T+pd.Timedelta(seconds=12)
    assert r.redBegin==T+pd.Timedelta(seconds=12) and r.redEnd==T+pd.Timedelta(seconds=19)


def test_quarantine_omits_entire_conflicting_cycle_and_preserves_other_cycles():
    e=events();e.loc[(e.eventParam==2)&(e.eventCode==8)&(e.timeStamp==T+pd.Timedelta(seconds=4)),'timeStamp']=T+pd.Timedelta(seconds=12)
    e=e[~((e.eventParam==2)&e.eventCode.isin([10,11])&(e.timeStamp<T+pd.Timedelta(seconds=20)))]
    p=build_vehicle_phase_profile(e,'S','2025-09-07')
    c=build_vehicle_cycle_profile(p,'S','2025-09-07',conflict_policy='quarantine')
    assert c.attrs['quarantined_cycle_count']==1
    assert c.cycleBegin.tolist()==[T+pd.Timedelta(seconds=30),T+pd.Timedelta(seconds=50)]
    validate_cycle_profile(c)
    assert c.greenPhase2.iloc[0]==[(T+pd.Timedelta(seconds=40),T+pd.Timedelta(seconds=44))]


def test_ambiguous_start_is_excluded_and_requires_fresh_barrier_transition():
    e=events();extra=pd.DataFrame([('S',T+pd.Timedelta(seconds=t),code,6) for t,code in [(30,1),(34,8),(35,10),(36,11)]],columns=e.columns)
    p=build_vehicle_phase_profile(pd.concat([e,extra],ignore_index=True),'S','2025-09-07')
    c=build_vehicle_cycle_profile(p,'S','2025-09-07',conflict_policy='quarantine')
    assert c.attrs['quarantined_cycle_count']>0
    assert all(not(a<=T+pd.Timedelta(seconds=30)<b) for a,b in zip(c.cycleBegin,c.cycleEnd))
