"""Deterministic batch checks and conversions with independent model outputs."""

import argparse
import os
import time
from pathlib import Path

from urdf2mjcf.inspection import inspect_urdf
from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.reporting import new_report, validate_report_path, write_report
from urdf2mjcf.urdf_to_mujoco_converter import (
    MujocoConversionManager,
    add_common_arguments,
)


class BatchConversionManager:
    """Process a directory of URDFs sequentially, retaining each model's report.

    By default, model failures do not stop subsequent jobs. ``run()`` returns
    an aggregate report, including discovery failures; inspect ``status`` or
    ``summary`` to determine whether all models succeeded.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        *,
        backend: str = "urdf2mjcf",
        fixed_base: bool = True,
        export_collision: bool = False,
        auto_recalculate_inertia: bool = True,
        recursive: bool = True,
        check_only: bool = False,
        fail_fast: bool = False,
    ):
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else self.input_dir.with_name(f"{self.input_dir.name}_mjcf")
        )
        self.backend = backend
        self.conversion_options = {
            "backend": backend,
            "fixed_base": fixed_base,
            "export_collision": export_collision,
            "auto_recalculate_inertia": auto_recalculate_inertia,
        }
        self.recursive = recursive
        self.check_only = check_only
        self.fail_fast = fail_fast
        self.last_report: dict | None = None
        self.logger = URDF2MJCFLogger.get_logger("BatchConversionManager")

    def _discover(self) -> list[tuple[Path, Path]]:
        """Plan all outputs before creating any directories."""
        if self.backend not in {"urdf2mjcf", "mujoco", "auto"}:
            raise ValueError(f"Unknown conversion backend: {self.backend}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Input directory does not exist: {self.input_dir}"
            )
        if not self.check_only and self.input_dir.is_relative_to(self.output_dir):
            raise ValueError(
                "Batch output must not equal or contain the input directory"
            )
        if (
            not self.check_only
            and self.output_dir.exists()
            and not self.output_dir.is_dir()
        ):
            raise NotADirectoryError(
                f"Output path is not a directory: {self.output_dir}"
            )

        def scan_error(error):
            raise error

        exclude_output = (
            self.output_dir != self.input_dir
            and self.output_dir.is_relative_to(self.input_dir)
        )

        def is_output(path):
            if not exclude_output:
                return False
            try:
                return path.resolve().is_relative_to(self.output_dir)
            except (OSError, RuntimeError):
                # A bad file symlink is a model failure, not a discovery failure.
                return False

        sources = []
        for current, children, files in os.walk(self.input_dir, onerror=scan_error):
            directory = Path(current)
            children[:] = sorted(
                name
                for name in children
                if self.recursive
                and not (directory / name).is_symlink()
                and not is_output(directory / name)
            )
            sources.extend(
                directory / name
                for name in files
                if Path(name).suffix.lower() == ".urdf"
                and not is_output(directory / name)
            )
        if not sources:
            raise ValueError(f"No URDF files found in: {self.input_dir}")
        if self.check_only:
            return [(source, self.output_dir) for source in sorted(sources)]

        jobs = []
        destinations = {}
        for source in sorted(sources):
            relative = source.relative_to(self.input_dir)
            destination = (self.output_dir / relative.with_suffix("")).resolve()
            if not destination.is_relative_to(self.output_dir):
                raise ValueError(
                    f"Model output escapes batch output directory: {destination}"
                )
            if destination in destinations:
                raise ValueError(
                    f"Conflicting model outputs: {destinations[destination]} and {source}"
                )
            destinations[destination] = source
            jobs.append((source, destination))
        # A model named a.urdf and another in a/... must not share asset trees.
        for destination, source in destinations.items():
            for parent in destination.parents:
                if parent in destinations:
                    raise ValueError(
                        f"Overlapping model outputs: {destinations[parent]} and {source}"
                    )
        self._validate_source_dependencies(jobs)
        return jobs

    def _validate_source_dependencies(self, jobs: list[tuple[Path, Path]]) -> None:
        """Protect every job's inputs from every other job's publication.

        A single-model manager can only check its own output. A batch must also
        reject references into another model's replaced asset tree before the
        first job runs, regardless of processing order or source symlinks.
        """
        asset_trees = {(destination / "meshes").resolve() for _, destination in jobs}
        resolved_sources = {}
        model_files = set()
        for source, destination in jobs:
            try:
                resolved = source.resolve()
            except (OSError, RuntimeError):
                continue  # The per-job error must not stop unrelated models.
            resolved_sources[source] = resolved
            model_files.add(destination / f"{resolved.stem}.xml")

        def protect(path: Path, kind: str, source: Path) -> None:
            if path in model_files or any(
                parent in asset_trees for parent in path.parents
            ):
                raise ValueError(
                    f"Batch output would replace an input {kind}: {path} (used by {source})"
                )

        for source, resolved in resolved_sources.items():
            protect(resolved, "URDF", source)
            # Inspection retains resolved references even if other declarations
            # are invalid. Such model errors still belong to per-job reports.
            for mesh in inspect_urdf(source, backend=self.backend)["meshes"]:
                if mesh["resolved_path"] is not None:
                    protect(Path(mesh["resolved_path"]), "mesh", source)

    def _run_one(self, source: Path, destination: Path) -> dict:
        started = time.perf_counter()
        report = new_report(
            "check" if self.check_only else "conversion", source, self.backend
        )
        manager = None
        try:
            if self.check_only:
                report = inspect_urdf(source, backend=self.backend)
                for issue in report["issues"]:
                    log = (
                        self.logger.warning
                        if issue["severity"] == "warning"
                        else self.logger.error
                    )
                    log("%s: [%s] %s", source, issue["code"], issue["message"])
            else:
                manager = MujocoConversionManager(
                    source, destination, **self.conversion_options
                )
                manager.run()
                report = manager.last_report
        except Exception as exc:
            if manager is not None and manager.last_report is not None:
                report = manager.last_report
            report.update(status="failed", error=str(exc))
            report.setdefault(
                "duration_seconds", round(time.perf_counter() - started, 6)
            )
            self.logger.error("%s failed: %s", source, exc)
            self.logger.debug("Model failure details", exc_info=True)
        report["relative_input"] = source.relative_to(self.input_dir).as_posix()
        report["output_directory"] = None if self.check_only else str(destination)
        return report

    def run(self) -> dict:
        started = time.perf_counter()
        report = new_report(
            "batch_check" if self.check_only else "batch_conversion",
            self.input_dir,
            self.backend,
        )
        self.last_report = report
        report.update(
            stage="discovery",
            output_directory=None if self.check_only else str(self.output_dir),
            options={
                **self.conversion_options,
                "recursive": self.recursive,
                "check_only": self.check_only,
                "fail_fast": self.fail_fast,
            },
            results=[],
            summary={"total": 0, "succeeded": 0, "failed": 0, "skipped": 0},
        )
        try:
            jobs = self._discover()
            report["summary"]["total"] = len(jobs)
            report["stage"] = "processing"
            for source, destination in jobs:
                if self.fail_fast and report["summary"]["failed"]:
                    result = {
                        "input": str(source),
                        "relative_input": source.relative_to(self.input_dir).as_posix(),
                        "status": "skipped",
                        "reason": "Stopped after an earlier model failed (--fail-fast)",
                        "output_directory": (
                            None if self.check_only else str(destination)
                        ),
                    }
                    report["summary"]["skipped"] += 1
                else:
                    self.logger.info(
                        "Processing %s", source.relative_to(self.input_dir)
                    )
                    result = self._run_one(source, destination)
                    counter = (
                        "succeeded"
                        if result["status"] in {"success", "passed"}
                        else "failed"
                    )
                    report["summary"][counter] += 1
                report["results"].append(result)
            report.update(
                status="failed" if report["summary"]["failed"] else "success",
                stage="completed",
            )
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            self.logger.error("Batch failed: %s", exc)
            self.logger.debug("Batch failure details", exc_info=True)
        finally:
            report["duration_seconds"] = round(time.perf_counter() - started, 6)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or convert a directory of URDF models with independent outputs."
    )
    parser.add_argument(
        "input_dir", help="Directory containing URDF files (scanned recursively)."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Output root; defaults to sibling <input_dir>_mjcf.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan URDFs directly inside input_dir.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed model and mark remaining models skipped.",
    )
    args = parser.parse_args()
    if args.verbose:
        URDF2MJCFLogger.set_level("DEBUG")
    logger = URDF2MJCFLogger.get_logger("BatchConversionManager")
    try:
        report_path = (
            validate_report_path(args.report, args.input_dir) if args.report else None
        )
    except (OSError, ValueError) as exc:
        logger.error("Invalid report path: %s", exc)
        return 1
    report = new_report(
        "batch_check" if args.check else "batch_conversion",
        args.input_dir,
        args.backend,
    )
    try:
        manager = BatchConversionManager(
            args.input_dir,
            args.output_dir,
            backend=args.backend,
            fixed_base=not args.floating_base,
            export_collision=args.export_collision,
            auto_recalculate_inertia=not args.no_inertia_recalc,
            check_only=args.check,
            recursive=not args.no_recursive,
            fail_fast=args.fail_fast,
        )
        report = manager.run()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        logger.error("Batch failed: %s", exc, exc_info=args.verbose)
    if report_path is not None:
        try:
            write_report(report, report_path)
            logger.info("Batch report saved to '%s'", report_path)
        except Exception as exc:
            logger.error("Cannot write batch report: %s", exc, exc_info=args.verbose)
            return 1
    summary = report.get("summary", {})
    logger.info(
        "Batch %s: %s succeeded, %s failed, %s skipped (%s total).",
        report["status"],
        summary.get("succeeded", 0),
        summary.get("failed", 0),
        summary.get("skipped", 0),
        summary.get("total", 0),
    )
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
