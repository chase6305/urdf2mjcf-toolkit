"""Versioned, atomic JSON reports for checks and conversions."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def _input_path(source: str | Path) -> Path:
    source = Path(source).expanduser()
    try:
        return source.resolve()
    except (OSError, RuntimeError):
        # Reporting must still work when the input itself is an unreadable or
        # cyclic symlink. Opening/validating it will supply the actual error.
        return source.absolute()


def new_report(kind: str, source: str | Path, backend: str) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "input": str(_input_path(source)),
        "requested_backend": backend,
        "status": "running",
        "stage": "initialization",
        "started_at": datetime.now(UTC).isoformat(),
    }


def validate_report_path(path: str | Path, source: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination == _input_path(source):
        raise ValueError("Report path must not overwrite the input file")
    if destination.suffix.lower() != ".json":
        raise ValueError("Report path must have a .json extension")
    if destination.exists() and not destination.is_file():
        raise ValueError("Report path must be a file")
    return destination


def write_report(report: dict, destination: str | Path) -> Path:
    """Replace a report only after complete JSON serialization and writing."""
    path = Path(destination).expanduser().resolve()
    contents = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as pending:
        pending_path = Path(pending.name)
        try:
            pending.write(contents)
            pending.close()
            pending_path.replace(path)
        finally:
            pending_path.unlink(missing_ok=True)
    return path
