"""Write-ahead admissions for bounded RNA experiments in the workflow journal."""

import hashlib
from typing import Any, cast

from .. import record_io
from . import journal
from .models import AutomatedWorkflowConfig


class CandidateBudgetExceeded(ValueError):
    """The next scientific experiment exceeds an explicit execution limit."""


def candidate_identity(inputs: dict[str, Any], *, graph: bool = False) -> str:
    """Identify exact numerical inputs, omitting resolution for graph reuse."""
    values = dict(inputs)
    parameters = dict(values["parameters"])
    parameters.pop("candidateId", None)
    if graph:
        parameters.pop("leidenResolution", None)
    values["parameters"] = parameters
    return hashlib.sha256(record_io.canonical_json_bytes(values)).hexdigest()


class CandidateBudget:
    """Reconstruct bounded admissions from fixed immutable journal slots."""

    def __init__(
        self,
        store: Any,
        prefix: str,
        workflow_run_id: str,
        config: AutomatedWorkflowConfig,
        provenance: dict[str, Any],
        *,
        previous_provenances: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.store = store
        self.prefix = prefix
        self.workflow_run_id = workflow_run_id
        self.config = config
        self.provenance = provenance
        self.admissions: dict[str, list[dict[str, Any]]] = {}
        for scope in ("sample0", "sample1", "full"):
            limit = (
                config.maxFullPartitions
                if scope == "full"
                else config.maxScreeningEvaluations
            )
            rows: list[dict[str, Any]] = []
            for slot in range(limit):
                record = journal.read_checkpoint(
                    store,
                    prefix,
                    workflow_run_id,
                    self._key(scope, slot, "admission"),
                )
                if record is None:
                    continue
                if record["inputs"] not in (provenance, *previous_provenances):
                    raise ValueError(
                        "Candidate admission has changed checkpoint inputs"
                    )
                row = record["outputs"]
                if (
                    row.get("slot") != slot
                    or row.get("scope") != scope
                    or slot != len(rows)
                ):
                    raise ValueError("Candidate admission history is inconsistent")
                rows.append(row)
            self.admissions[scope] = rows
        self.repairs = self._read_repairs(previous_provenances)

    @staticmethod
    def _repair_inputs(
        cells: dict[str, Any], setting: dict[str, Any], experiment: dict[str, Any]
    ) -> dict[str, Any]:
        """Identify a scientific intervention independently of review attempts."""
        baseline = dict(setting)
        baseline["parameters"] = {
            key: value
            for key, value in setting["parameters"].items()
            if key != "candidateId"
        }
        return {"cells": cells, "baseline": baseline, "experiment": experiment}

    def _read_repairs(
        self, previous_provenances: tuple[dict[str, Any], ...]
    ) -> dict[str, dict[str, Any]]:
        """Derive reserved repairs from this journal, including earlier decisions."""
        repairs: dict[str, dict[str, Any]] = {}
        explicit_admission = False
        repair_keys = [
            f"parameter_tuning/full/repairs/{slot}"
            for slot in range(self.config.maxFullRepairs + 1)
        ]
        scopes = {"parameter_tuning/full"} | {
            "parameter_tuning/evidence_revisions/"
            + hashlib.sha256(record_io.canonical_json_bytes(provenance)).hexdigest()
            + "/full"
            for provenance in (self.provenance, *previous_provenances)
        }
        review_keys = [
            f"{scope}/review{index}{suffix}"
            for scope in sorted(scopes)
            for index in range(self.config.maxFullPartitions + 1)
            for suffix in ("", "/answer")
        ]
        for key in [*repair_keys, *review_keys]:
            record = journal.read_checkpoint(
                self.store, self.prefix, self.workflow_run_id, key
            )
            if record is None:
                continue
            if key in repair_keys:
                inputs = record["inputs"]
                identity = hashlib.sha256(
                    record_io.canonical_json_bytes(inputs)
                ).hexdigest()
                if record["outputs"] != {"identity": identity, "reserved": True}:
                    raise ValueError("Full-cohort repair admission is inconsistent")
                repairs[identity] = inputs
                explicit_admission = True
                continue
            # At most one repair is supported. Once explicitly admitted, later
            # unadmitted proposals must not masquerade as additional work.
            if explicit_admission:
                continue
            evidence, action = record["inputs"], record["outputs"]["action"]
            if (
                action.get("action") != "experiment"
                or evidence.get("comparisonCoverage", {}).get("phase") != "validation"
            ):
                continue
            experiment = evidence["experiments"][action["experimentId"]]
            if experiment["parameter"] == "useHarmony":
                continue
            candidate_id = action["selectedCandidateId"]
            candidate = next(
                row
                for row in evidence["candidates"]
                if row["candidateId"] == candidate_id
            )
            inputs = self._repair_inputs(
                candidate["cellSelection"],
                evidence["settings"][candidate_id],
                experiment,
            )
            identity = hashlib.sha256(
                record_io.canonical_json_bytes(inputs)
            ).hexdigest()
            repairs[identity] = inputs
        if len(repairs) > self.config.maxFullRepairs:
            raise CandidateBudgetExceeded(
                "Saved full-cohort repairs exceed the configured allowance"
            )
        return repairs

    def admit_repair(
        self, cells: dict[str, Any], setting: dict[str, Any], experiment: dict[str, Any]
    ) -> None:
        """Reserve one global full-cohort repair before executing its intervention."""
        inputs = self._repair_inputs(cells, setting, experiment)
        identity = hashlib.sha256(record_io.canonical_json_bytes(inputs)).hexdigest()
        if identity in self.repairs:
            return
        if len(self.repairs) >= self.config.maxFullRepairs:
            raise CandidateBudgetExceeded(
                "The allowed full-cohort repair has been used; scientific acceptance remains unresolved"
            )
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            f"parameter_tuning/full/repairs/{len(self.repairs)}",
            inputs,
            {"identity": identity, "reserved": True},
        )
        self.repairs[identity] = inputs

    def admission_provenance(self, admission: dict[str, Any]) -> dict[str, Any]:
        """Return the verified original inputs without changing an admission."""
        record = journal.read_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(admission["scope"], admission["slot"], "admission"),
        )
        if record is None or record["outputs"] != admission:
            raise ValueError("Candidate admission is missing or changed")
        return cast(dict[str, Any], record["inputs"])

    @staticmethod
    def _key(scope: str, slot: int, kind: str) -> str:
        return f"parameter_tuning/{scope}/evaluation{slot}/{kind}"

    def admit(self, scope: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if scope not in self.admissions:
            raise ValueError("Unknown candidate execution scope")
        identity = candidate_identity(inputs)
        rows = self.admissions[scope]
        for row in rows:
            if row["identity"] == identity:
                return row
        graph_identity = candidate_identity(inputs, graph=True)
        if scope == "full":
            if len(rows) >= self.config.maxFullPartitions:
                raise CandidateBudgetExceeded("Full-cohort partition limit reached")
            graphs = {row["graphIdentity"] for row in rows} | {graph_identity}
            if len(graphs) > self.config.maxFullGraphs:
                raise CandidateBudgetExceeded("Full-cohort graph limit reached")
        else:
            if len(rows) >= self.config.maxScreeningEvaluations:
                raise CandidateBudgetExceeded("Screening candidate limit reached")
            total = sum(len(self.admissions[name]) for name in ("sample0", "sample1"))
            if total >= self.config.maxTotalScreeningEvaluations:
                raise CandidateBudgetExceeded("Total screening candidate limit reached")
        row = {
            "slot": len(rows),
            "scope": scope,
            "identity": identity,
            "graphIdentity": graph_identity,
            "executionInputs": inputs,
        }
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(scope, len(rows), "admission"),
            inputs=self.provenance,
            outputs=row,
        )
        rows.append(row)
        return row

    def check_many(
        self, scope: str, inputs: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Check room for a comparison without committing an admission."""
        existing = self.admissions[scope]
        identities = {row["identity"] for row in existing}
        pending = {
            candidate_identity(value): value
            for value in inputs
            if candidate_identity(value) not in identities
        }
        total = len(existing) + len(pending)
        if scope == "full":
            graphs = {row["graphIdentity"] for row in existing} | {
                candidate_identity(value, graph=True) for value in pending.values()
            }
            if (
                total > self.config.maxFullPartitions
                or len(graphs) > self.config.maxFullGraphs
            ):
                raise CandidateBudgetExceeded(
                    "The full-cohort comparison exceeds the remaining graph/partition limit"
                )
        elif (
            total > self.config.maxScreeningEvaluations
            or sum(len(self.admissions[name]) for name in ("sample0", "sample1"))
            + len(pending)
            > self.config.maxTotalScreeningEvaluations
        ):
            raise CandidateBudgetExceeded(
                "The screening comparison exceeds the remaining candidate limit"
            )
        return pending

    def admit_many(self, scope: str, inputs: list[dict[str, Any]]) -> None:
        """Check a baseline or matched pair in full before its first computation."""
        for value in self.check_many(scope, inputs).values():
            self.admit(scope, value)

    def completed(self, admission: dict[str, Any]) -> dict[str, Any] | None:
        return journal.load_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(admission["scope"], admission["slot"], "complete"),
            inputs=admission,
        )

    def completed_source(
        self, inputs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Find exact completed evidence before charging additional validation work."""
        identity = candidate_identity(inputs)
        for rows in self.admissions.values():
            for admission in rows:
                if admission["identity"] == identity:
                    completed = self.completed(admission)
                    if completed is not None:
                        return admission, completed
        return None

    def complete(self, admission: dict[str, Any], output: dict[str, Any]) -> None:
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow_run_id,
            self._key(admission["scope"], admission["slot"], "complete"),
            inputs=admission,
            outputs=output,
        )

    def summary(self) -> dict[str, Any]:
        """Count distinct reserved and completed comparisons, including reuse."""
        scopes = {}
        for scope, rows in self.admissions.items():
            completed = [row for row in rows if self.completed(row) is not None]
            scopes[scope] = {
                state: {
                    "graphs": len({row["graphIdentity"] for row in entries}),
                    "partitions": len({row["identity"] for row in entries}),
                }
                for state, entries in (("reserved", rows), ("completed", completed))
            }
        return {
            "scopes": scopes,
            "limits": {
                "perScreen": self.config.maxScreeningEvaluations,
                "totalScreens": self.config.maxTotalScreeningEvaluations,
                "fullPartitions": self.config.maxFullPartitions,
                "fullGraphs": self.config.maxFullGraphs,
            },
        }
