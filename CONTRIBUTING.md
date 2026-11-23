# Contributing

## Repository language

Use English for repository documentation, code comments and docstrings, command
help and diagnostic messages, dashboard text, test descriptions, commit messages,
and pull request descriptions.

Preserve identifiers, source records, place names, and reference data in their
original form. Do not translate data values to satisfy the documentation language
policy. Local operational logs and investigation artifacts belong in the ignored
data/scratch directories, not in Git.

## Validation

Install with the dependencies in `pyproject.toml` and run:

```bash
python -m pytest tests -q
```

For extraction changes, also compare representative real outputs against saved
reference data and document intentional schema or metadata changes. Tests and
investigations must use isolated output directories rather than publish into
operational results. See [checked pipeline operations](docs/checked_pipeline.md)
for completion rules and quality checks.
