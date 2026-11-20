"""Validate lane configuration without inventing operational mappings.

Parsing is cached by SHA-256, rules version and the selected phase/barrier map.
The returned frame uses the names consumed by the extractor package. Decisions are evaluated
outside that cache because an exclusion can apply to only some data dates.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import pandas as pd

RULES_VERSION = "config-validation-v2-mapping-evidence"
# Historical default phase-to-barrier map; not reviewed controller evidence.
DEFAULT_BARRIER_MAP = {1: 2, 2: 2, 5: 2, 6: 2, 3: 1, 4: 1, 7: 1, 8: 1}
_ALIASES = {
    "id": "ID", "approach": "approach", "lanetype": "laneType",
    "width": "width", "length": "length", "lanenofromleft": "laneNoFromLeft",
    "type": "type", "channel": "channelNo", "channelno": "channelNo",
    "phase": "phaseNo", "phaseno": "phaseNo", "stopbardistance": "stopBarDistance",
    "slotnumber": "slotNo", "slotno": "slotNo", "countbar": "countBar",
    "barrier": "barrierNo", "barrierno": "barrierNo",
    "pedestrianphase": "pedestrianPhaseNo", "pedestrianphaseno": "pedestrianPhaseNo",
}
_NULLS = {"", "null", "none", "nan", "na", "n/a"}
_CACHE: OrderedDict[tuple, "ConfigValidation"] = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_LIMIT = 4096


@dataclass
class ConfigValidation:
    status: str
    reason_code: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    config_hash: str | None = None
    dataframe: pd.DataFrame | None = field(default=None, repr=False)
    capabilities: dict[str, Any] = field(default_factory=dict)
    action: str = "repair_pending"
    signal_id: str | None = None
    path: str | None = None
    rules_version: str = RULES_VERSION
    decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe audit record; the potentially large frame stays in memory."""
        return {
            "status": self.status, "reason_code": self.reason_code,
            "issues": copy.deepcopy(self.issues), "config_hash": self.config_hash,
            "capabilities": copy.deepcopy(self.capabilities), "action": self.action,
            "signal_id": self.signal_id, "path": self.path,
            "rules_version": self.rules_version, "decision": copy.deepcopy(self.decision),
            "row_count": 0 if self.dataframe is None else len(self.dataframe),
            "columns": [] if self.dataframe is None else list(self.dataframe.columns),
        }


def clear_validation_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _issue(code: str, message: str, *, severity: str = "error", **details) -> dict:
    return {"reason_code": code, "message": message, "severity": severity, **details}


def _rows(mask: pd.Series) -> list[int]:
    # CSV header is line 1; blank lines are discarded by the parser.
    return [int(i) + 2 for i in mask[mask].index]


def _canonical(header: str) -> str:
    stripped = header.strip()
    return _ALIASES.get(re.sub(r"[\s_.-]+", "", stripped).lower(), stripped)


def _capability(applicable, valid, reason_code, required_columns):
    return {"applicable": applicable, "valid": bool(valid), "reason_code": reason_code,
            "required_columns": list(required_columns)}


def _capabilities(frame: pd.DataFrame, issues: list[dict]) -> dict:
    errors = {i["reason_code"] for i in issues if i["severity"] == "error"}
    phase_ok = "phaseNo" in frame and frame["phaseNo"].notna().any()
    vehicle = _capability(True, phase_ok, "valid" if phase_ok else "invalid_phase_mapping", ["phaseNo"])
    # Lane inventories do not prove that pedestrian signals are absent.
    pedestrian = _capability(None, phase_ok, "pedestrian_applicability_unknown", ["phaseNo"])
    caps = {"vehicle_profiles": vehicle, "pedestrian_profiles": pedestrian}
    required = ["phaseNo", "channelNo", "stopBarDistance", "laneType"]
    missing = [c for c in required if c not in frame]
    detector_errors = errors & {"invalid_channel", "missing_channel_values", "missing_phase_values",
        "missing_detector_values", "invalid_detector_values", "conflicting_channel_mapping",
        "invalid_phase_mapping", "invalid_barrier_mapping"}
    for kind in ("front", "back", "count"):
        if missing:
            caps[kind] = _capability(None, False, "missing_detector_columns", required)
            caps[kind]["missing_columns"] = missing
            continue
        # Restrict row-specific detector faults to the detector class that uses
        # those rows. An unknown distance/count flag affects every possible class.
        if kind == "count":
            relevant = (frame["countBar"].isna() | frame["countBar"].eq(1)) if "countBar" in frame else pd.Series(True, index=frame.index)
        else:
            distances = frame["stopBarDistance"]
            relevant = distances.isna() | (distances.eq(0) if kind == "front" else distances.gt(0))
        relevant_rows = set(_rows(relevant))
        scoped_errors = {i["reason_code"] for i in issues
                         if i["severity"] == "error" and i["reason_code"] in detector_errors
                         and (not i.get("rows") or relevant_rows.intersection(i["rows"]))}
        if scoped_errors:
            caps[kind] = _capability(None, False, sorted(scoped_errors)[0], required)
            continue
        detectors = frame[frame["channelNo"].notna() & frame["phaseNo"].notna()]
        if kind == "count":
            if "countBar" not in detectors:
                caps[kind] = _capability(None, False, "count_detector_applicability_unknown", required + ["countBar"])
                continue
            selected = detectors[detectors["countBar"] == 1]
        else:
            mask = detectors["stopBarDistance"].eq(0) if kind == "front" else detectors["stopBarDistance"].gt(0)
            selected = detectors[mask]
        # A complete detector table supports a negative capability; an empty one
        # is an incomplete inventory, not evidence of operational absence.
        applicable = bool(len(selected)) if len(detectors) else None
        caps[kind] = _capability(applicable, True, "valid" if applicable else
            ("no_configured_" + kind + "_detector" if applicable is False else "detector_applicability_unknown"), required)
    feature_kinds = {
        "vehicle_phase_profile": "vehicle_profiles", "vehicle_cycle_profile": "vehicle_profiles",
        "pedestrian_phase_profile": "pedestrian_profiles", "pedestrian_cycle_profile": "pedestrian_profiles",
        "spat": "vehicle_profiles", "volume": "back", "headway": "back", "gap": "front", "pet": "back",
        "shockwave": "back", "occupancy": "front", "split_failure": "front",
        "red_running": "front", "dilemma_running": "front",
        "pedestrian_activity": "pedestrian_profiles", "pedestrian_delay": "pedestrian_profiles",
        "conflict": "back", "conflict_headway": "back", "conflict_pet": "back", "conflict_propensity": "back",
    }
    mapping_warnings = [copy.deepcopy(i) for i in issues if i["reason_code"] == "ambiguous_channel_mapping"]
    for kind in ("front", "back", "count"):
        caps[kind]["mapping_review_required"] = bool(mapping_warnings)
        caps[kind]["mapping_issues"] = mapping_warnings
    caps["features"] = {name: copy.deepcopy(caps[kind]) for name, kind in feature_kinds.items()}
    return caps


def _parse_and_validate(raw: bytes, mapping: dict[int, int]) -> ConfigValidation:
    issues = []
    if not raw:
        return ConfigValidation("invalid", "empty_config", [_issue("empty_config", "Configuration is 0 bytes.")])
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return ConfigValidation("invalid", "csv_parse_error", [_issue("csv_parse_error", str(exc))])
    if not text.strip():
        return ConfigValidation("invalid", "whitespace_config", [_issue("whitespace_config", "Configuration contains only whitespace.")])
    try:
        rows = [row for row in csv.reader(io.StringIO(text), strict=True) if any(cell.strip() for cell in row)]
    except csv.Error as exc:
        return ConfigValidation("invalid", "csv_parse_error", [_issue("csv_parse_error", str(exc))])
    if not rows:
        return ConfigValidation("invalid", "empty_column_name", [_issue("empty_column_name", "CSV contains no named columns or values.")])
    header, records = rows[0], rows[1:]
    if any(len(row) != len(header) for row in records):
        return ConfigValidation("invalid", "csv_parse_error", [_issue("csv_parse_error", "CSV row width differs from header.")])
    columns = []
    seen = {}
    for original in header:
        canonical = _canonical(original)
        seen[canonical] = seen.get(canonical, 0) + 1
        if not canonical:
            issues.append(_issue("empty_column_name", "CSV contains an empty header."))
        if seen[canonical] > 1:
            # Original lane files contain lane Length and detector Length. Keep
            # both positions as pandas does; never conflate their meanings.
            if canonical == "length":
                canonical = f"length.{seen[canonical] - 1}"
                issues.append(_issue("duplicate_length_preserved", "Repeated Length column retained separately.", severity="info"))
            else:
                issues.append(_issue("duplicate_column", f"Headers resolve to the same column: {canonical}."))
                canonical = f"{canonical}.{seen[canonical] - 1}"
        columns.append(canonical)
    if columns != header:
        issues.append(_issue("headers_normalized", "Header whitespace and known aliases normalized.", severity="info", original=header, normalized=columns))
    frame = pd.DataFrame(records, columns=columns)
    for col in frame:
        frame[col] = frame[col].map(lambda value: pd.NA if value.strip().lower() in _NULLS else value.strip())
    if frame.empty:
        issues.insert(0, _issue("header_only_config", "CSV has a header but no configuration rows."))
    if "phaseNo" not in frame:
        issues.append(_issue("missing_phase_column", "Vehicle profiles require Phase/phaseNo.", columns=["phaseNo"]))
    for col, reason in [("phaseNo", "invalid_phase_mapping"), ("channelNo", "invalid_channel"), ("barrierNo", "invalid_barrier_mapping"), ("pedestrianPhaseNo", "invalid_phase_mapping")]:
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        present = frame[col].notna()
        invalid = present & (values.isna() | values.le(0) | values.mod(1).ne(0))
        if invalid.any():
            issues.append(_issue(reason, f"{col} must be a positive integer; operational meaning is validated separately.",
                column=col, rows=_rows(invalid), values=frame.loc[invalid, col].astype(str).unique().tolist()))
        # Retain the offending source values in issues, never coerce them into a
        # guessed valid phase. Invalid rows get NA in a frame that cannot be used.
        frame[col] = values.mask(invalid).astype("Int64")
    if "phaseNo" in frame and not frame["phaseNo"].notna().any() and not frame.empty:
        issues.append(_issue("missing_phase_values", "No usable vehicle phase is configured."))
    if {"barrierNo", "phaseNo"}.issubset(frame):
        mismatch = frame["barrierNo"].notna() & frame["phaseNo"].notna() & frame["barrierNo"].ne(frame["phaseNo"].map(mapping))
        if mismatch.any():
            issues.append(_issue("invalid_barrier_mapping", "Barrier contradicts the selected phase/barrier map.", rows=_rows(mismatch)))
    if "stopBarDistance" in frame:
        values = pd.to_numeric(frame["stopBarDistance"], errors="coerce")
        invalid = frame["stopBarDistance"].notna() & (values.isna() | values.lt(0) | ~values.abs().lt(float("inf")))
        if invalid.any():
            issues.append(_issue("invalid_detector_values", "Stopbar distance must be finite and nonnegative.", rows=_rows(invalid)))
        frame["stopBarDistance"] = values.mask(invalid)
    if "countBar" in frame:
        values = pd.to_numeric(frame["countBar"], errors="coerce")
        invalid = frame["countBar"].notna() & ~values.isin([0, 1])
        if invalid.any():
            issues.append(_issue("invalid_detector_values", "countBar must be 0 or 1.", rows=_rows(invalid)))
        frame["countBar"] = values.mask(invalid).astype("Int64")
    detector_columns = [c for c in ("channelNo", "type", "stopBarDistance") if c in frame]
    detector_rows = frame[detector_columns].notna().any(axis=1) if detector_columns else pd.Series(False, index=frame.index)
    if detector_rows.any():
        required = ["phaseNo", "channelNo", "stopBarDistance", "laneType"]
        missing = [c for c in required if c not in frame]
        if missing:
            issues.append(_issue("missing_detector_columns", "Configured detectors lack feature-required columns.", columns=missing))
        for col in required:
            if col not in frame:
                continue
            invalid = detector_rows & frame[col].isna()
            if invalid.any():
                reason = {"phaseNo": "missing_phase_values", "channelNo": "missing_channel_values"}.get(col, "missing_detector_values")
                issues.append(_issue(reason, f"Configured detector row lacks {col}.", column=col, rows=_rows(invalid)))
    if "channelNo" in frame:
        for channel, group in frame.dropna(subset=["channelNo"]).groupby("channelNo"):
            if len(group) < 2:
                continue
            # Connected loops can intentionally share a channel across lanes,
            # phases and distances. Flag that ambiguity without declaring the
            # complete inventory invalid. A stable lane ID assigning the same
            # channel to contradictory lane identities is a separate error.
            ambiguous = any(col in group and group[col].nunique() > 1
                            for col in ("approach", "stopBarDistance"))
            conflicting = False
            if "ID" in group:
                for _, same_lane in group.dropna(subset=["ID"]).groupby("ID"):
                    if any(col in same_lane and same_lane[col].nunique() > 1
                           for col in ("approach", "laneNoFromLeft")):
                        conflicting = True
            if conflicting:
                issues.append(_issue("conflicting_channel_mapping", "The same lane ID and channel have contradictory lane identities.", channel=int(channel), rows=[int(i)+2 for i in group.index]))
            elif ambiguous:
                issues.append(_issue("ambiguous_channel_mapping", "A channel spans approaches or detector distances; grouped detector semantics need source review.", severity="warning", channel=int(channel), rows=[int(i)+2 for i in group.index]))
    caps = _capabilities(frame, issues)
    errors = [i for i in issues if i["severity"] == "error"]
    fatal = {"empty_column_name", "duplicate_column", "header_only_config", "missing_phase_column"}
    usable = "phaseNo" in frame and frame["phaseNo"].notna().any()
    usable = usable and not any(i["reason_code"] in fatal for i in errors)
    return ConfigValidation("invalid" if errors else "valid", errors[0]["reason_code"] if errors else "valid",
        issues, dataframe=frame, capabilities=caps, action="use" if usable else "repair_pending")


def _date(value: Any) -> date:
    if isinstance(value, date):
        return date.fromisoformat(value.isoformat()[:10])
    return date.fromisoformat(str(value))


def _apply_decision(result: ConfigValidation, decision: Mapping | None, data_date) -> None:
    if decision is None:
        return
    result.decision = dict(decision)
    action = decision.get("action", "repair_pending")
    if action not in {"use", "repair_pending", "exclude"}:
        error = "Unknown action."
    elif action != "exclude":
        if action == "repair_pending":
            result.action = action
        return
    else:
        error = None
        source = decision.get("source") or decision.get("config_source")
        if not all(isinstance(v, str) and v.strip() for v in (decision.get("reason"), source, decision.get("signal_id"))):
            error = "Exclusion requires reason, signal_id and configuration source."
        elif str(decision["signal_id"]) != result.signal_id:
            error = "Exclusion targets a different signal."
        else:
            try:
                start, end, target = _date(decision["start_date"]), _date(decision["end_date"]), _date(data_date)
                if start > end:
                    error = "Exclusion date range is reversed."
                elif not start <= target <= end:
                    result.issues.append(_issue("exclusion_outside_period", "Exclusion does not apply to this data date.", severity="info"))
                    return
            except (ValueError, TypeError, KeyError):
                error = "Exclusion requires ISO start_date/end_date and a data_date."
        if error is None:
            result.action = "exclude"
            result.issues.append(_issue("explicit_exclusion", decision["reason"], severity="info", source=source))
            return
    result.status = "invalid"
    result.reason_code = "invalid_exclusion_decision"
    result.action = "repair_pending"
    result.issues.append(_issue("invalid_exclusion_decision", error))


def validate_config(path, *, signal_id=None, barrier_map=None, decision=None, data_date=None) -> ConfigValidation:
    """Validate a CSV and optional dated exclusion; never modify the source file.

    ``barrier_map`` maps each allowed positive integer phase to its positive
    integer barrier. ``decision`` accepts action, signal_id, reason, source,
    start_date, end_date. Explicit exclusions require the current data_date.
    A valid status alone does not establish every feature's applicability; use
    ``capabilities['features']`` when planning feature-specific work.
    """
    path = Path(path)
    selected = DEFAULT_BARRIER_MAP if barrier_map is None else barrier_map
    try:
        mapping = {}
        for phase, barrier in selected.items():
            phase_num, barrier_num = float(phase), float(barrier)
            if phase_num <= 0 or barrier_num <= 0 or not phase_num.is_integer() or not barrier_num.is_integer():
                raise ValueError("Phase/barrier keys and values must be positive integers.")
            if int(phase_num) in mapping and mapping[int(phase_num)] != int(barrier_num):
                raise ValueError("Conflicting entries for the same phase.")
            mapping[int(phase_num)] = int(barrier_num)
        if not mapping:
            raise ValueError("The phase/barrier map is empty.")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        result = ConfigValidation("invalid", "invalid_barrier_mapping", [_issue("invalid_barrier_mapping", str(exc))])
    else:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            result = ConfigValidation("missing", "missing_config", [_issue("missing_config", "Configuration file does not exist.")])
        except OSError as exc:
            result = ConfigValidation("invalid", "unreadable_config", [_issue("unreadable_config", str(exc))])
        else:
            digest = hashlib.sha256(raw).hexdigest()
            key = (digest, RULES_VERSION, json.dumps(mapping, sort_keys=True))
            with _CACHE_LOCK:
                cached = _CACHE.get(key)
                if cached is None:
                    cached = _parse_and_validate(raw, mapping)
                    cached.config_hash = digest
                    _CACHE[key] = cached
                    if len(_CACHE) > _CACHE_LIMIT:
                        _CACHE.popitem(last=False)
                _CACHE.move_to_end(key)
                result = copy.deepcopy(cached)
    result.rules_version = RULES_VERSION
    result.signal_id = str(signal_id) if signal_id is not None else path.stem
    result.path = str(path)
    _apply_decision(result, decision, data_date)
    return result
