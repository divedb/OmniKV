"""The support matrix must stay accurate against the code (research.md §9
M5: "machine-readable support matrix shipped with the release").

Validates docs/support_matrix.json structurally and cross-checks its
enum coverage against the implementation: every StateKind and every
layout family must appear with a status the legend defines, and any
status the legend claims must be used. This turns "the matrix is
accurate" into a checkable invariant rather than a promise.
"""

from __future__ import annotations

import json
import os

import pytest

from omnikv.types import FORMAT_VERSION, LAYOUT_FAMILIES, StateKind

MATRIX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
    "support_matrix.json",
)


@pytest.fixture(scope="module")
def matrix() -> dict:
    with open(MATRIX_PATH) as f:
        return json.load(f)


def _sections(matrix: dict) -> dict[str, dict[str, dict]]:
    return {
        name: section
        for name, section in matrix.items()
        if isinstance(section, dict)
        and isinstance(next(iter(section.values()), None), dict)
        and all(
            isinstance(v, dict) and "status" in v for v in section.values()
        )
    }


class TestSupportMatrix:
    def test_file_exists_and_parses(self, matrix):
        assert matrix["matrix_version"] >= 1

    def test_format_version_matches_code(self, matrix):
        assert matrix["omnikv_format_version"] == FORMAT_VERSION, (
            "the matrix must track the FORMAT_VERSION it describes"
        )

    def test_every_state_kind_covered_exactly(self, matrix):
        kinds = {k.value for k in StateKind}
        covered = set(matrix["state_kinds"])
        assert covered == kinds, (
            f"state_kinds mismatch: missing={kinds - covered}, "
            f"unknown={covered - kinds}"
        )

    def test_every_layout_family_covered_exactly(self, matrix):
        covered = set(matrix["layouts"])
        assert covered == set(LAYOUT_FAMILIES), (
            f"layouts mismatch: missing={set(LAYOUT_FAMILIES) - covered}, "
            f"unknown={covered - set(LAYOUT_FAMILIES)}"
        )

    def test_all_statuses_are_declared(self, matrix):
        legend = set(matrix["status_legend"])
        used = {
            entry["status"]
            for section in _sections(matrix).values()
            for entry in section.values()
        }
        assert used <= legend, f"statuses missing from legend: {used - legend}"
        assert legend == used, (
            f"legend statuses never used (stale legend): {legend - used}"
        )

    def test_entries_carry_evidence_or_notes(self, matrix):
        for section_name, section in _sections(matrix).items():
            for name, entry in section.items():
                assert "evidence" in entry or "notes" in entry or "limits" in entry, (
                    f"{section_name}.{name} has no evidence, notes or limits"
                )

    def test_recurrent_stays_rejected(self, matrix):
        # The M4 no-go must not silently regress in the matrix.
        assert matrix["state_kinds"]["recurrent_state"]["status"] == "rejected"

    def test_block_outermost_kernel_split_stays_rejected(self, matrix):
        # The M5 silent-corruption closure must stay recorded as rejected.
        assert (
            matrix["kernel_block_splits"]["ratio_gt1_block_outermost"]["status"]
            == "rejected"
        )
