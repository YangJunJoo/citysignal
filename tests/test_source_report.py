import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from run_pipeline import source_report


def raw(root, d, timestamp):
    path = root / f'atspm-{d}.parquet'
    pq.write_table(pa.table({'timeStamp':[timestamp], 'signalID':['1001'], 'eventCode':[81], 'eventParam':[1]}), path)
    return path


def args(tmp_path, selected):
    directory = tmp_path/'raw'
    directory.mkdir(exist_ok=True)
    return SimpleNamespace(raw_dir=str(directory), processed_dir=str(tmp_path/'processed'),
                           jobs_file=None,quality_report=None,start_date=None,end_date=None, selected_dates=selected)


def test_selected_dates_preserve_neighbor_evidence_and_scoped_counts(tmp_path):
    options = args(tmp_path,['2026-08-26'])
    raw(Path(options.raw_dir),'2026-8-26','08-26-2026 06:10:59.900')
    neighbor=raw(Path(options.raw_dir),'2026-8-27','08-27-2026 00:00:00.000')
    report, _ = source_report(options,tmp_path/'run')
    assert len(report['files']) == 1
    assert len(report['context_files']) == 2
    assert report['reason_counts'] == {'adjacent_gap':1}
    assert report['status'] == 'partial'
    assert str(neighbor.resolve()) in [e['path'] for e in report['context_files']]


def test_publication_frontier_uses_full_inventory_without_scanning_it(tmp_path):
    options=args(tmp_path,['2026-09-10'])
    raw(Path(options.raw_dir),'2026-9-6','09-06-2026 19:59:59.900')
    (Path(options.raw_dir)/'atspm-2026-9-10.csv.download.json').write_text(json.dumps({'http_status':404}))
    report,_=source_report(options,tmp_path/'run')
    assert report['missing'][0]['status'] == 'source_pending'
    assert report['publication_frontier'] == '2026-09-06'
    assert report['context_files'] == []


def test_changed_neighbor_rejects_stale_supplied_report(tmp_path):
    options=args(tmp_path,['2026-08-26'])
    raw(Path(options.raw_dir),'2026-8-26','08-26-2026 23:59:59.900')
    neighbor=raw(Path(options.raw_dir),'2026-8-27','08-27-2026 00:00:00')
    _,path=source_report(options,tmp_path/'run')
    options.quality_report=path
    raw(Path(options.raw_dir),'2026-8-27','08-27-2026 02:00:00')
    with pytest.raises(ValueError,match='Stale raw-source quality report'):
        source_report(options,tmp_path/'second')


def test_empty_selected_dates_do_not_expand_to_full_inventory(tmp_path):
    options=args(tmp_path,[])
    raw(Path(options.raw_dir),'2026-8-26','08-26-2026 00:00:00')
    report,_=source_report(options,tmp_path/'run')
    assert report['status'] == 'no_work'
    assert not report['files'] and not report['accepted_paths']


def test_new_neighbor_rejects_stale_supplied_report(tmp_path):
    options=args(tmp_path,['2026-08-26'])
    raw(Path(options.raw_dir),'2026-8-26','08-26-2026 23:59:59.900')
    _,path=source_report(options,tmp_path/'run')
    options.quality_report=path
    raw(Path(options.raw_dir),'2026-8-27','08-27-2026 05:00:00')
    with pytest.raises(ValueError,match='unassessed adjacent/selected files'):
        source_report(options,tmp_path/'second')
