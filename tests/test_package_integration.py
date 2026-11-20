"""Regression coverage for the source-only extractor package and report tools."""

from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from feature_extractor.checked_execution import _execute_cycle_frames
from feature_extractor.config import ConfigParser
from feature_extractor.feature_registry import output_path
from feature_extractor.profiles import (build_pedestrian_phase_profile,
                                        build_pedestrian_cycle_profile)
from data_downloader.signal_config import load_config_summary, scrape_config_report
from scripts.compare_reference_outputs import main as compare_outputs


def test_entry_points_import_without_old_packages_or_browser_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = '''
import importlib.abc
import sys
class RejectOldPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'citysignal', 'citysignal_v2', 'selenium', 'webdriver_manager'}:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, RejectOldPackages())
sys.path.insert(0, sys.argv[1])
import feature_extractor.pipeline
import feature_extractor.checked_runner
import feature_extractor.process_local_atspm
import data_downloader.download_signal_configs
import data_downloader.update_config_summary
import run_pipeline
assert not any('/citysignal/Lib/' in path for path in sys.path)
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(root)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_has_one_extraction_implementation():
    parser = ConfigParser.create_parser()
    options = parser._option_string_actions
    assert '--use-v2' not in options and '--no-v2' not in options


def test_real_profiles_and_spat_are_published_without_vendor_wrapper(tmp_path):
    origin = pd.Timestamp('2025-09-07')
    rows = []
    for phase, offset in ((2, 0), (4, 10)):
        for start in (0, 20, 40, 60):
            for code, delta in ((1, 0), (8, 4), (10, 5), (11, 6)):
                rows.append(('S', origin + pd.Timedelta(seconds=start + offset + delta), code, phase))
    events = pd.DataFrame(rows, columns=['signalID', 'timeStamp', 'eventCode', 'eventParam'])
    validation = SimpleNamespace(dataframe=pd.DataFrame({'phaseNo': [2, 4]}), capabilities={})
    results = _execute_cycle_frames('S', '2025-09-07', events, validation, tmp_path, ['spat'])
    assert all(result.status == 'success' for result in results.values())
    actual = read_frame(output_path(tmp_path, 'spat', 'S', '2025-09-07'))
    assert len(actual) == 3
    assert actual.greenDurationPhase2.tolist() == [4.0, 4.0, 4.0]
    assert 'state_evidence' not in results['vehicle_cycle_profile'].data.attrs
    assert not list((tmp_path / '.staging').iterdir())


def test_pedestrian_sequences_preserve_incomplete_observations_and_cycle_boundaries():
    origin = pd.Timestamp('2025-09-07')
    raw = pd.DataFrame({
        'timeStamp': [origin + pd.Timedelta(seconds=x) for x in (0, 1, 2, 3, 5, 6, 7, 10, 11, 12)],
        'eventCode': [22, 21, 22, 23, 21, 22, 23, 21, 22, 23],
        'eventParam': [2] * 10})
    phases = build_pedestrian_phase_profile(raw, 'S', '2025-09-07')
    assert phases.correctSequenceFlag.tolist() == [0, 1, 1, 1]
    cycles = pd.DataFrame({'cycleNo': [1, 2],
        'cycleBegin': [origin, origin + pd.Timedelta(seconds=5)],
        'cycleEnd': [origin + pd.Timedelta(seconds=5), origin + pd.Timedelta(seconds=10)]})
    result = build_pedestrian_cycle_profile(phases, cycles, 'S', '2025-09-07')
    assert result.cycleNo.tolist() == [1, 2]
    assert result.pedestrianWalkBegin.tolist() == [origin + pd.Timedelta(seconds=x) for x in (1, 5)]


def test_config_summary_uses_explicit_or_configured_source(tmp_path, monkeypatch):
    path = tmp_path / 'config_summary.csv'
    path.write_text('signalID,siiaID\n0012,34\n')
    monkeypatch.setenv('ATSPM_CONFIG_SUMMARY_PATH', str(path))
    assert load_config_summary(converters={'signalID': str}).signalID.tolist() == ['0012']
    assert load_config_summary(path).siiaID.tolist() == [34]


def test_report_tables_are_saved_directly_and_browser_is_closed(tmp_path):
    pytest.importorskip('bs4')

    class Driver:
        page_source = '''<table id="lanes"><thead><tr><th>Phase</th><th>Channel</th></tr></thead>
        <tbody><tr><th>2</th><td>1</td></tr></tbody></table>'''
        closed = False
        def get(self, url):
            self.url = url
        def quit(self):
            self.closed = True

    driver = Driver()
    saved = scrape_config_report('34', 'S', tmp_path, driver_factory=lambda: driver, wait_time=0)
    assert saved == [tmp_path / 'lanes' / 'S.csv']
    assert pd.read_csv(saved[0]).to_dict('list') == {'Phase': [2], 'Channel': [1]}
    assert driver.closed and driver.url.endswith('id=34')
    assert not list(tmp_path.rglob('*.tmp'))


def test_report_parser_failure_closes_browser(tmp_path):
    pytest.importorskip('bs4')
    driver = SimpleNamespace(page_source='<table id="lanes"></table>',
                             get=lambda url: None, closed=False)
    driver.quit = lambda: setattr(driver, 'closed', True)
    with pytest.raises(ValueError, match='Incomplete'):
        scrape_config_report('34', 'S', tmp_path, driver_factory=lambda: driver, wait_time=0)
    assert driver.closed


def test_reference_comparison_is_read_only_and_checks_nested_values(tmp_path):
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    paths = [output_path(root, 'headway', 'S', '2025-09-07') for root in (reference, candidate)]
    for path in paths:
        path.parent.mkdir(parents=True)
        write_frame(pd.DataFrame({'samples': [[1.0, float('nan')]]}), path)
    argv = ['--reference-dir', str(reference), '--candidate-dir', str(candidate),
            '--signal', 'S', '--date', '2025-09-07', '--features', 'headway']
    before = {path: path.read_bytes() for path in paths}
    assert compare_outputs(argv) == 0
    assert {path: path.read_bytes() for path in paths} == before
    write_frame(pd.DataFrame({'samples': [[2.0, float('nan')]]}), paths[1])
    assert compare_outputs(argv) == 1
    paths[1].unlink()
    assert compare_outputs(argv) == 1
