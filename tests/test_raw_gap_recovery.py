from scripts.investigate_raw_gaps import compare_candidate

FIELDS = {'signalid':'string','timestamp':'string','eventcode':'int64','eventparam':'int64'}


def report(begin, end, counts):
    return {'source_date':'2026-08-26','status':'valid','min_timestamp':begin,'max_timestamp':end,
            'hourly_counts':counts,'row_count':sum(counts.values()),'signals':{},'schema_fields':FIELDS}


def test_identical_incomplete_remote_cannot_be_recoverable():
    local = report('2026-08-26T00:00:00','2026-08-26T06:10:59.900',{'00':10})
    following = report('2026-08-27T00:00:00','2026-08-27T23:59:59',{'00':20})
    result = compare_candidate(local,dict(local),following=following)
    assert result['conclusion'] == 'source_still_incomplete'
    assert result['same_observed_statistics']
    assert result['candidate_boundary_gaps'][0]['seconds'] == 64140.1
    assert not result['replacement_performed']


def test_candidate_must_close_known_gap_and_preserve_counts():
    local = report('2026-08-26T00:00:00','2026-08-26T06:10:59.900',{'00':10})
    candidate = report('2026-08-26T00:00:00','2026-08-26T23:59:59',{'00':10,'07':40})
    following = report('2026-08-27T00:00:00','2026-08-27T23:59:59',{'00':20})
    assert compare_candidate(local,candidate,following=following)['conclusion'] == 'recoverable'
    candidate['hourly_counts']['00'] = 9
    assert compare_candidate(local,candidate,following=following)['conclusion'] == 'source_still_incomplete'


def test_schema_mismatch_prevents_candidate_eligibility():
    local = report('2026-08-26T00:00:00','2026-08-26T06:10:59.900',{'00':10})
    candidate = report('2026-08-26T00:00:00','2026-08-26T23:59:59',{'00':10,'07':40})
    candidate['schema_fields'] = {'timestamp':'string'}
    result = compare_candidate(local,candidate)
    assert not result['schema_compatible']
    assert result['conclusion'] == 'source_still_incomplete'


def test_cached_candidate_does_not_claim_current_source_without_remote_version(tmp_path,monkeypatch):
    import json
    import scripts.investigate_raw_gaps as investigation
    output=tmp_path/'report';stage=output/'staging';stage.mkdir(parents=True)
    candidate=stage/'atspm-2026-8-26.csv';candidate.write_bytes(b'old')
    candidate.with_name(candidate.name+'.download.json').write_text(json.dumps({'status':'downloaded','byte_count':3,'etag':'"old"'}))
    monkeypatch.setattr(investigation,'probe',lambda url:{'url':url,'http_status':None,'error':'timeout'})
    investigation.main(['--dates','2026-08-26','--raw-dir',str(tmp_path/'raw'),'--output-dir',str(output)])
    result=json.loads((output/'recovery_investigation.json').read_text())['results'][0]
    assert result['conclusion'] == 'unavailable'
    assert 'No complete validated candidate' in result['error']
