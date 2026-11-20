"""Canonical output/dependency contract for extraction, reuse and reporting."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    path: str
    dependencies: tuple[str, ...] = ()
    scope: str = 'feature'
    required: bool = True


def _profile(kind, traffic, deps=()):
    return FeatureSpec(f'{traffic}_{kind}_profile',
                       f'signal_profile/{kind}/{traffic}_signal', deps, 'profile')


_specs = [
    _profile('phase', 'vehicle'),
    _profile('phase', 'pedestrian'),
    _profile('cycle', 'vehicle', ('vehicle_phase_profile',)),
    _profile('cycle', 'pedestrian', ('pedestrian_phase_profile', 'vehicle_cycle_profile')),
]
for _name in ('spat', 'volume', 'headway', 'gap', 'pet', 'occupancy', 'red_running',
              'dilemma_running', 'shockwave', 'pedestrian_activity', 'pedestrian_delay',
              'split_failure', 'conflict', 'conflict_pet', 'conflict_propensity'):
    _ped = _name.startswith('pedestrian_')
    _deps = ('pedestrian_cycle_profile',) if _ped else ('vehicle_cycle_profile',)
    _deps += {'split_failure': ('spat', 'occupancy'), 'conflict': ('headway',),
              'conflict_pet': ('pet',),
              'conflict_propensity': ('volume', 'pedestrian_activity')}.get(_name, ())
    _group = 'vehicle_signal' if _name == 'spat' else (
        'pedestrian_traffic' if _ped or _name == 'conflict_propensity' else 'vehicle_traffic')
    _specs.append(FeatureSpec(_name, f'feature/cycle/{_group}/{_name.removeprefix("pedestrian_")}', _deps, required=_name != 'conflict_propensity'))

FEATURES = {spec.name: spec for spec in _specs}
FEATURE_ORDER = tuple(FEATURES)
ALIASES = {'conflict_headway': 'conflict'}


def resolve_features(names=None):
    requested = [name for name, spec in FEATURES.items() if spec.required] if names is None else list(names)
    if not requested:
        raise ValueError('At least one feature must be requested')
    selected = set()
    def add(name):
        name = ALIASES.get(name, name)
        if name not in FEATURES:
            raise ValueError(f'Unknown feature: {name}')
        if name not in selected:
            selected.add(name)
            for dependency in FEATURES[name].dependencies:
                add(dependency)
    for name in requested:
        add(name)
    return tuple(name for name in FEATURE_ORDER if name in selected)


def output_path(output_dir, name, signal_id, date_str):
    return Path(output_dir) / 'fdot_d5' / 'feature_extraction' / FEATURES[name].path / str(signal_id) / f'{date_str}.parquet'
