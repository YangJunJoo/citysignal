"""Config validation tests use small inventories, never operational outputs."""
import json
from unittest.mock import patch

import pytest

from feature_extractor import config_validator as cv


def config(tmp_path, content, name="1234.csv"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("content, reason", [
    ("", "empty_config"), (" \n\t\r\n", "whitespace_config"),
    ("Phase,Channel\n", "header_only_config"),
    ('Phase,Channel\n2,"unterminated', "csv_parse_error"),
    ("Phase,Channel\n2,4,extra\n", "csv_parse_error"),
    ("ID,Approach\n1,North\n", "missing_phase_column"),
    (",,,\n,,,\n", "empty_column_name"),
])
def test_distinguishes_unusable_files(tmp_path, content, reason):
    result = cv.validate_config(config(tmp_path, content))
    assert result.status == "invalid"
    assert result.reason_code == reason
    assert result.action == "repair_pending"
    json.dumps(result.to_dict())


def test_missing_file(tmp_path):
    result = cv.validate_config(tmp_path / "absent.csv")
    assert (result.status, result.reason_code) == ("missing", "missing_config")


def test_normalizes_headers_values_and_preserves_two_lengths(tmp_path):
    path = config(tmp_path, " Phase , Channel ,Stopbar Distance, Lane Type ,Length,Length\n 2 , 4 , 0 , Through , 100 , 6 \n")
    before = path.read_bytes()
    result = cv.validate_config(path)
    assert result.status == "valid"
    assert result.dataframe.loc[0, "phaseNo"] == 2
    assert result.dataframe.loc[0, "laneType"] == "Through"
    assert result.dataframe.loc[0, "length"] == "100"
    assert result.dataframe.loc[0, "length.1"] == "6"
    assert path.read_bytes() == before


@pytest.mark.parametrize("phase", ["8R", "2.5", "-1", "0", "inf"])
def test_invalid_phase_is_never_guessed(tmp_path, phase):
    result = cv.validate_config(config(tmp_path, f"Phase\n{phase}\n"))
    assert result.reason_code == "invalid_phase_mapping"
    assert result.dataframe["phaseNo"].isna().all()


def test_custom_barrier_map_and_declared_barrier(tmp_path):
    path = config(tmp_path, "Phase,Barrier\n9,3\n")
    assert cv.validate_config(path).status == "valid"  # Syntax only; this is not operational evidence.
    assert cv.validate_config(path, barrier_map={9: 3}).status == "valid"
    assert cv.validate_config(path, barrier_map={9: 1}).reason_code == "invalid_barrier_mapping"
    assert cv.validate_config(path, barrier_map={9.5: 1}).reason_code == "invalid_barrier_mapping"


@pytest.mark.parametrize("channel", ["text", "2.5", "-1", "0"])
def test_invalid_channel(tmp_path, channel):
    result = cv.validate_config(config(tmp_path, f"Phase,Channel,Stopbar Distance,Lane Type\n2,{channel},0,Through\n"))
    assert result.reason_code == "invalid_channel"


def test_phase_only_config_keeps_profiles_without_inventing_applicability(tmp_path):
    result = cv.validate_config(config(tmp_path, "Phase\n2\n6\n"))
    assert result.status == "valid"
    assert result.capabilities["vehicle_profiles"]["valid"]
    assert result.capabilities["pedestrian_profiles"]["applicable"] is None
    assert result.capabilities["front"]["applicable"] is None
    assert not result.capabilities["front"]["valid"]


def test_feature_required_columns_and_no_front_detector(tmp_path):
    invalid = cv.validate_config(config(tmp_path, "Phase,Channel\n2,4\n"))
    assert invalid.reason_code == "missing_detector_columns"
    valid = cv.validate_config(config(tmp_path, "Phase,Channel,Stopbar Distance,Lane Type\n2,4,150,Through\n"))
    assert valid.status == "valid"
    assert valid.capabilities["features"]["volume"]["applicable"] is True
    assert valid.capabilities["features"]["occupancy"]["applicable"] is False
    assert valid.capabilities["features"]["gap"]["applicable"] is False
    assert valid.capabilities["features"]["pet"]["applicable"] is True


def test_inventory_lanes_without_detectors_do_not_invalidate_other_rows(tmp_path):
    result = cv.validate_config(config(tmp_path,
        "ID,Phase,Channel,Stopbar Distance,Lane Type\n1,2,4,0,Through\n2,,,,Right\n"))
    assert result.status == "valid"


def test_permitted_one_to_many_vs_contradictory_channel_mapping(tmp_path):
    header = "ID,Approach,Lane No. from Left,Phase,Channel,Stopbar Distance,Lane Type\n"
    rows = "1,North,1,2,4,0,Through\n2,North,2,2,4,0,Through\n1,North,1,5,4,0,Left\n"
    assert cv.validate_config(config(tmp_path, header + rows)).status == "valid"
    conflicting = rows + "1,North,1,2,4,150,Through\n"
    multiple_distances = cv.validate_config(config(tmp_path, header + conflicting))
    assert multiple_distances.status == "valid"
    assert multiple_distances.capabilities["front"]["mapping_review_required"]
    conflicting = rows + "3,South,1,6,4,0,Through\n"
    assert cv.validate_config(config(tmp_path, header + conflicting)).status == "valid"
    conflicting = rows + "1,South,2,6,4,0,Through\n"
    assert cv.validate_config(config(tmp_path, header + conflicting)).reason_code == "conflicting_channel_mapping"


def test_cache_keys_content_rules_map_and_returns_independent_frames(tmp_path):
    cv.clear_validation_cache()
    path = config(tmp_path, "Phase\n2\n")
    with patch.object(cv, "_parse_and_validate", wraps=cv._parse_and_validate) as parse:
        first = cv.validate_config(path)
        first.dataframe.loc[0, "phaseNo"] = 5
        assert cv.validate_config(path).dataframe.loc[0, "phaseNo"] == 2
        assert parse.call_count == 1
        path.write_text("Phase\n6\n")
        assert cv.validate_config(path).config_hash != first.config_hash
        assert parse.call_count == 2
        cv.validate_config(path, barrier_map={6: 1})
        assert parse.call_count == 3
        with patch.object(cv, "RULES_VERSION", "different-rules"):
            cv.validate_config(path)
        assert parse.call_count == 4


def test_exclusion_requires_scoped_evidence_and_is_evaluated_per_date(tmp_path):
    path = config(tmp_path, "")
    decision = dict(action="exclude", signal_id="1234", reason="Configuration unavailable per operator review",
                    source="review/1234.json", start_date="2026-08-26", end_date="2026-09-06")
    result = cv.validate_config(path, decision=decision, data_date="2026-09-01")
    assert result.action == "exclude"
    assert result.status == "invalid"  # exclusion does not repair the file
    assert cv.validate_config(path, decision=decision, data_date="2026-09-07").action == "repair_pending"
    for missing in ("signal_id", "reason", "source", "start_date", "end_date"):
        incomplete = {key: value for key, value in decision.items() if key != missing}
        assert cv.validate_config(path, decision=incomplete, data_date="2026-09-01").reason_code == "invalid_exclusion_decision"
    assert cv.validate_config(path, decision=decision).reason_code == "invalid_exclusion_decision"
    assert cv.validate_config(path, decision={**decision, "signal_id": "different"}, data_date="2026-09-01").action == "repair_pending"


def test_decision_cannot_make_invalid_config_usable(tmp_path):
    assert cv.validate_config(config(tmp_path, ""), decision={"action": "use"}).action == "repair_pending"


def test_ambiguous_alias_headers_rejected(tmp_path):
    assert cv.validate_config(config(tmp_path, "Phase,phaseNo\n2,6\n")).reason_code == "duplicate_column"
