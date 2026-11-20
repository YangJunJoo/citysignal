"""
pipeline.py — Unified feature extraction orchestrator for ATSPM.

Production calculation: extract_features_checked().
Legacy DataFrame/auto-save API: extract_all_features().

Manages:
  1. Data loading (single SignalDataCache per signal-day)
  2. Dependency ordering (e.g., conflict depends on headway, split_failure depends on spat + occupancy)
  3. Feature extraction execution with proper error handling
  4. Aggregation of results
"""

import time
import logging
import traceback
import pandas as pd

from feature_extractor.feature_registry import FEATURES, resolve_features
from feature_extractor.result_types import FeatureResult

from .data_loader import SignalDataCache
from .extractors.spat import extract_spat
from .extractors.volume import extract_volume
from .extractors.headway import extract_headway
from .extractors.gap import extract_gap
from .extractors.occupancy import extract_occupancy
from .extractors.red_running import extract_red_running, extract_dilemma_running
from .extractors.shockwave import extract_shockwave
from .extractors.pedestrian import extract_pedestrian_activity, extract_pedestrian_delay
from .extractors.secondary import extract_split_failure, extract_conflict, extract_conflict_headway, extract_conflict_pet, extract_conflict_propensity



class _CalculationCache:
    """Delegate cached reads while withholding the extractors' legacy writes."""

    def __init__(self, cache):
        self._cache = cache

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def save_feature(self, df, event_type, feature_name):
        # Persistence belongs to the common executor after result validation.
        return None


def extract_features_checked(cache, features, prior_results=None,
                             applicability=None) -> dict[str, FeatureResult]:
    """Calculate selected features and dependencies without writing outputs.

    Profiles must be supplied by the common executor in ``prior_results``. A
    successful prior result must contain its DataFrame; files are never silently
    loaded here. Only explicit ``(False, reason)`` applicability evidence creates
    a ``not_applicable`` outcome. Unknown applicability still runs extraction.
    Every selected dependency is returned, and an unsuccessful dependency blocks
    its consumers while independent calculations continue.
    """
    selected = resolve_features(features)
    prior_results = {} if prior_results is None else prior_results
    applicability = {} if applicability is None else applicability
    results = {}
    calculation_cache = _CalculationCache(cache)

    extractors = {
        'spat': lambda: extract_spat(calculation_cache),
        'volume': lambda: extract_volume(calculation_cache),
        'headway': lambda: extract_headway(calculation_cache),
        'gap': lambda: extract_gap(calculation_cache),
        'pet': lambda: extract_gap(calculation_cache, with_stopbar=False),
        'occupancy': lambda: extract_occupancy(calculation_cache),
        'red_running': lambda: extract_red_running(calculation_cache),
        'dilemma_running': lambda: extract_dilemma_running(calculation_cache),
        'shockwave': lambda: extract_shockwave(calculation_cache),
        'pedestrian_activity': lambda: extract_pedestrian_activity(calculation_cache),
        'pedestrian_delay': lambda: extract_pedestrian_delay(calculation_cache),
        'split_failure': lambda: extract_split_failure(
            calculation_cache, results['spat'].data, results['occupancy'].data),
        'conflict': lambda: extract_conflict(calculation_cache, results['headway'].data),
        'conflict_pet': lambda: extract_conflict_pet(calculation_cache, results['pet'].data),
        'conflict_propensity': lambda: extract_conflict_propensity(
            calculation_cache, results['volume'].data, results['pedestrian_activity'].data),
    }

    for name in selected:
        applies, reason = applicability.get(name, (None, ''))
        if applies is False:
            if isinstance(reason, str) and reason.strip():
                results[name] = FeatureResult(
                    name, 'not_applicable', 'explicit_not_applicable', reason)
            else:
                results[name] = FeatureResult(
                    name, 'blocked', 'applicability_unverified',
                    'Not-applicable classification requires an explicit evidence reason.')
            continue

        if name in prior_results:
            prior = prior_results[name]
            if not isinstance(prior, FeatureResult) or prior.name != name:
                results[name] = FeatureResult(
                    name, 'failed', 'invalid_prior_result',
                    f'Prior result must be a FeatureResult for {name}.')
            elif prior.status == 'success' and not isinstance(prior.data, pd.DataFrame):
                results[name] = FeatureResult(
                    name, 'failed', 'invalid_prior_result',
                    'Successful prior result must include a pandas DataFrame.')
            elif prior.status == 'success' and prior.data.empty:
                results[name] = FeatureResult(
                    name, 'no_data', 'empty_prior_result',
                    'Prior result contains an empty DataFrame.', data=prior.data)
            else:
                results[name] = prior
            continue

        dependencies = FEATURES[name].dependencies
        unavailable = [dependency for dependency in dependencies
                       if dependency not in results or results[dependency].status != 'success']
        if unavailable:
            details = '; '.join(
                f'{dependency}: {results[dependency].status} '
                f'({results[dependency].reason_code}) {results[dependency].message}'
                if dependency in results else f'{dependency}: missing'
                for dependency in unavailable)
            states = {results[dependency].status for dependency in unavailable if dependency in results}
            reason_code = ('dependency_failed' if states & {'failed', 'blocked'} else
                           'dependency_no_data' if 'no_data' in states else
                           'dependency_not_applicable' if 'not_applicable' in states else
                           'dependency_skipped' if 'skipped' in states else 'missing_dependency')
            results[name] = FeatureResult(name, 'blocked', reason_code, details)
            continue

        if FEATURES[name].scope == 'profile':
            results[name] = FeatureResult(
                name, 'blocked', 'missing_profile_result',
                'Profile results must be provided by the common executor.')
            continue

        try:
            frame = extractors[name]()
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f'{name} returned {type(frame).__name__}; expected pandas DataFrame')
            from .profiles import apply_state_quality
            frame = apply_state_quality(frame, cache.df_vehicle_cycle, name) if hasattr(cache, 'df_vehicle_cycle') else frame
            if frame.empty:
                results[name] = FeatureResult(
                    name, 'no_data', frame.attrs.get('reason_code', 'empty_result'),
                    frame.attrs.get('message', 'Extractor returned an empty DataFrame.'), data=frame)
            else:
                results[name] = FeatureResult(name, 'success', data=frame, row_count=len(frame))
        except Exception:
            message = traceback.format_exc()
            logging.error('  %s FAILED: %s', name, message)
            results[name] = FeatureResult(name, 'failed', 'extractor_exception', message)

    return results


def extract_all_features(signal_id: str, year: int, month: int, day: int,
                          cwd: str, relative_interim: str, relative_production: str,
                          purdue_standard: bool = True,
                          with_countbar: bool = False,
                          features: list = None,
                          suppress_errors: bool = True,
                          config_dir: str = None) -> dict:
    """
    Extract all features for a single signal-day using the vectorized engine.
    
    This replaces the 14 separate calls to extract_* methods in the v1 pipeline,
    providing a single data load and proper dependency management.
    
    Parameters
    ----------
    signal_id : str
        Signal identifier
    year, month, day : int
        Date components
    cwd : str
        Working directory root
    relative_interim : str
        Relative path to interim data
    relative_production : str
        Relative path to production data
    purdue_standard : bool
        Split failure threshold mode
    with_countbar : bool
        Detector filter mode for volume/red_running
    features : list, optional
        Specific features to extract. If None, extracts all.
        Available: 'spat', 'volume', 'headway', 'gap', 'pet', 'occupancy',
                   'red_running', 'dilemma_running', 'shockwave',
                   'pedestrian_activity', 'pedestrian_delay',
                   'split_failure', 'conflict', 'conflict_propensity'
    suppress_errors : bool
        If True, catch and log errors per feature instead of failing entirely
    
    Returns
    -------
    dict
        Dictionary of {feature_name: DataFrame} for each extracted feature.
        Failed features will have empty DataFrames.
    """
    t_start = time.time()
    
    # Step 1: Single data load
    cache = SignalDataCache(
        signal_id=signal_id,
        year=year, month=month, day=day,
        cwd=cwd,
        relative_interim=relative_interim,
        relative_production=relative_production,
        config_dir=config_dir
    )
    
    t_load = time.time()
    logging.info(f"Cached data load: {t_load - t_start:.2f}s")
    
    # Define feature set
    all_features = [
        'spat', 'volume', 'headway', 'gap', 'pet', 'occupancy',
        'red_running', 'dilemma_running', 'shockwave',
        'pedestrian_activity', 'pedestrian_delay',
        'split_failure', 'conflict', 'conflict_headway', 'conflict_pet', 'conflict_propensity'
    ]
    
    if features is None:
        features = all_features
    
    results = {}
    
    # Step 2: Extract primary features (no dependencies)
    primary_extractors = {
        'spat': lambda: extract_spat(cache),
        'volume': lambda: extract_volume(cache, with_countbar=with_countbar),
        'headway': lambda: extract_headway(cache),
        'gap': lambda: extract_gap(cache),
        'pet': lambda: extract_gap(cache, with_stopbar=False),
        'occupancy': lambda: extract_occupancy(cache),
        'red_running': lambda: extract_red_running(cache, with_countbar=with_countbar),
        'dilemma_running': lambda: extract_dilemma_running(cache, with_countbar=with_countbar),
        'shockwave': lambda: extract_shockwave(cache),
        'pedestrian_activity': lambda: extract_pedestrian_activity(cache),
        'pedestrian_delay': lambda: extract_pedestrian_delay(cache),
    }
    
    for feature_name, extractor in primary_extractors.items():
        if feature_name not in features:
            continue
        try:
            t_feat = time.time()
            results[feature_name] = extractor()
            elapsed = time.time() - t_feat
            logging.info(f"  {feature_name}: {elapsed:.2f}s, {len(results[feature_name])} rows")
        except Exception as e:
            if suppress_errors:
                logging.error(f"  {feature_name} FAILED: {e}")
                results[feature_name] = pd.DataFrame()
            else:
                raise
    
    # Step 3: Extract secondary features (depend on primary results)
    if 'split_failure' in features:
        try:
            t_feat = time.time()
            df_spat = results.get('spat', pd.DataFrame())
            df_occupancy = results.get('occupancy', pd.DataFrame())
            
            # If not extracted yet, need to extract dependencies
            if len(df_spat) == 0 and 'spat' not in results:
                df_spat = extract_spat(cache)
            if len(df_occupancy) == 0 and 'occupancy' not in results:
                df_occupancy = extract_occupancy(cache)
            
            results['split_failure'] = extract_split_failure(
                cache, df_spat, df_occupancy, purdue_standard=purdue_standard
            )
            elapsed = time.time() - t_feat
            logging.info(f"  split_failure: {elapsed:.2f}s, {len(results['split_failure'])} rows")
        except Exception as e:
            if suppress_errors:
                logging.error(f"  split_failure FAILED: {e}")
                results['split_failure'] = pd.DataFrame()
            else:
                raise
    
    for conflict_name in ('conflict', 'conflict_headway'):
        if conflict_name not in features:
            continue
        try:
            t_feat = time.time()
            df_headway = results.get('headway', pd.DataFrame())
            
            if len(df_headway) == 0 and 'headway' not in results:
                df_headway = extract_headway(cache)
            
            extractor = extract_conflict if conflict_name == 'conflict' else extract_conflict_headway
            results[conflict_name] = extractor(cache, df_headway)
            elapsed = time.time() - t_feat
            logging.info(f"  {conflict_name}: {elapsed:.2f}s, {len(results[conflict_name])} rows")
        except Exception as e:
            if suppress_errors:
                logging.error(f"  {conflict_name} FAILED: {e}")
                results[conflict_name] = pd.DataFrame()
            else:
                raise
    
    if 'conflict_pet' in features:
        try:
            t_feat = time.time()
            df_pet = results.get('pet', pd.DataFrame())
            
            if len(df_pet) == 0 and 'pet' not in results:
                df_pet = extract_gap(cache, with_stopbar=False)
            
            results['conflict_pet'] = extract_conflict_pet(cache, df_pet)
            elapsed = time.time() - t_feat
            logging.info(f"  conflict_pet: {elapsed:.2f}s, {len(results['conflict_pet'])} rows")
        except Exception as e:
            if suppress_errors:
                logging.error(f"  conflict_pet FAILED: {e}")
                results['conflict_pet'] = pd.DataFrame()
            else:
                raise
    
    if 'conflict_propensity' in features:
        try:
            t_feat = time.time()
            df_volume = results.get('volume', pd.DataFrame())
            df_ped = results.get('pedestrian_activity', pd.DataFrame())
            
            results['conflict_propensity'] = extract_conflict_propensity(cache, df_volume, df_ped)
            elapsed = time.time() - t_feat
            logging.info(f"  conflict_propensity: {elapsed:.2f}s, {len(results['conflict_propensity'])} rows")
        except Exception as e:
            if suppress_errors:
                logging.error(f"  conflict_propensity FAILED: {e}")
                results['conflict_propensity'] = pd.DataFrame()
            else:
                raise
    
    total_elapsed = time.time() - t_start
    logging.info(f"Total extraction for {signal_id}: {total_elapsed:.2f}s, "
                 f"{sum(len(df) for df in results.values())} total rows across {len(results)} features")
    
    return results
