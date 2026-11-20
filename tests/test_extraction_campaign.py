import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_extraction_campaign as campaign
from feature_extractor.checked_runner import discover_jobs


def test_priority_dates_and_boundary_sources_are_exact():
    targets,boundaries,sources=campaign.priority_plan()
    assert len(targets)==72 and len(set(targets))==72
    assert boundaries==['2024-06-01','2024-10-16','2025-02-08','2025-09-16']
    assert len(sources)==76 and sources==sorted(sources)
    assert not set(targets).intersection(boundaries)


def test_campaign_attempts_priority_then_newest_and_keeps_partial(tmp_path,monkeypatch):
    directory=tmp_path/'campaign'
    output=tmp_path/'new-output'
    monkeypatch.setattr(campaign,'priority_plan',lambda:(['2024-05-01','2024-05-02'],[],['2024-05-01','2024-05-02']))
    monkeypatch.setattr(campaign,'newest_plan',lambda raw,exclude:['2026-11-19','2026-11-18'])
    monkeypatch.setattr(campaign.signal,'signal',lambda *args:None)
    calls=[]
    class Child:
        pid=1
        def __init__(self,cmd,**kwargs):
            calls.append(cmd)
            self.returncode=2 if len(calls)==1 else 0
        def poll(self):return self.returncode
    monkeypatch.setattr(campaign.subprocess,'Popen',Child)
    monkeypatch.setattr('sys.argv',['campaign','--campaign-dir',str(directory),'--processed-dir',str(output)])
    campaign.main()
    assert [cmd[cmd.index('--start-date')+1] for cmd in calls]==['2024-05-01','2024-05-02','2026-11-19','2026-11-18']
    assert all('--overwrite-extract' not in cmd and '--lock-held' in cmd for cmd in calls)
    state=json.loads((directory/'state.json').read_text())
    assert state['status']=='finished_with_unresolved'
    assert state['date_counts']=={'partial':1,'complete':3}
    campaign.main()  # Resuming completed date attempts does not run them again.
    assert len(calls)==4


def test_campaign_refuses_existing_operational_output(tmp_path,monkeypatch):
    monkeypatch.setattr('sys.argv',['campaign','--campaign-dir',str(tmp_path/'campaign'),
                                   '--processed-dir','/data2/ATSPM/processed_data'])
    with pytest.raises(SystemExit):campaign.main()


def test_single_day_discovery_handles_padded_names_without_touching_other_dates(tmp_path,monkeypatch):
    directory=tmp_path/'S';directory.mkdir()
    (directory/'atspm-2025-09-07.parquet').touch()
    (directory/'atspm-2025-9-8.parquet').touch()
    seen=[]
    def job(path,features):
        seen.append(path.name)
        return {'date_str':'2025-09-07'}
    monkeypatch.setattr('feature_extractor.checked_runner.job_from_path',job)
    config=SimpleNamespace(event_dir=str(tmp_path),signals=None,year=None,month=None,day=None,
                           features=['spat'],start_date='2025-09-07',end_date='2025-09-07',auto_mode=False)
    assert len(discover_jobs(config))==1
    assert seen==['atspm-2025-09-07.parquet']
