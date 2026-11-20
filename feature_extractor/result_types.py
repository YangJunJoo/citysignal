"""Serializable outcomes shared by both extraction engines and their consumers."""
from dataclasses import dataclass
from typing import Any

RESULT_STATUSES = frozenset({'success', 'failed', 'no_data', 'not_applicable', 'blocked', 'skipped'})
RULE_VERSION = 'checked-extraction-5-mapping-evidence'


@dataclass
class FeatureResult:
    name: str
    status: str
    reason_code: str = ''
    message: str = ''
    data: Any = None
    output_file_path: str | None = None
    row_count: int = 0

    def __post_init__(self):
        if self.status not in RESULT_STATUSES:
            raise ValueError(f'Unknown feature status: {self.status}')

    def to_dict(self):
        return {k: getattr(self, k) for k in ('name', 'status', 'reason_code', 'message',
                                             'output_file_path', 'row_count')}


def completion_status(results, required):
    """Absence and empty outputs cannot satisfy a required result."""
    return 'complete' if all(
        name in results and results[name].status in {'success', 'not_applicable'}
        for name in required
    ) else 'partial'
