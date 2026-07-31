import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from scarf.quality_control.hto import hto_demux

_FIXTURE_PATH = Path(__file__).parent / "seurat_hto_5_5_1_golden.json"


def _golden_fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_FIXTURE_PATH.read_text()))


def test_seurat_hto_golden_fixture_pins_its_provenance() -> None:
    fixture = _golden_fixture()
    provenance = fixture["provenance"]
    inputs = fixture["inputs"]

    assert provenance["package"] == "Seurat"
    assert provenance["packageVersion"] == "5.5.1"
    assert provenance["seuratObjectVersion"] == "5.4.0"
    assert provenance["fitdistrplusVersion"] == "1.2.6"
    assert provenance["dataset"] == "GSE245108"
    assert provenance["sourceFile"] == (
        "GSE245108_TNC-1-2-1_filtered_feature_bc_matrix.h5"
    )
    assert provenance["sourceSha256"] == (
        "22d64482c0fb04a7685e641d762efdd95a6498156ac22e7b0264925b0907b589"
    )
    assert provenance["nCells"] == 1000
    assert provenance["nHtos"] == 10
    assert provenance["cellSampleSeed"] == 20260731
    assert provenance["randomSeed"] == 0
    assert provenance["normalization"] == {"method": "CLR", "margin": 1}
    assert provenance["clustering"] == {
        "method": "kmeans",
        "clusterCount": 11,
        "nStarts": 100,
    }
    assert provenance["cutoff"] == {
        "distribution": "negative_binomial",
        "positiveQuantile": 0.99,
        "comparison": "strictly_greater",
    }
    assert provenance["referenceFunction"] == "Seurat::HTODemux"

    fingerprint_values = {
        "barcodes": inputs["barcodes"],
        "htoNames": inputs["htoNames"],
        "rawCounts": inputs["rawCounts"],
    }
    encoded = json.dumps(
        fingerprint_values,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == inputs["sha256"]


def test_hto_demux_matches_seurat_5_5_1_golden_calls() -> None:
    fixture = _golden_fixture()
    provenance = fixture["provenance"]
    inputs = fixture["inputs"]
    expected = fixture["seurat"]
    counts = pd.DataFrame(
        inputs["rawCounts"],
        index=inputs["barcodes"],
        columns=inputs["htoNames"],
        dtype=np.int64,
    )

    assignments = hto_demux(counts, random_seed=provenance["randomSeed"])

    np.testing.assert_array_equal(
        assignments.to_numpy(),
        np.asarray(expected["hashId"]),
    )
    global_assignments = np.where(
        assignments.isin(["Negative", "Doublet"]),
        assignments,
        "Singlet",
    )
    np.testing.assert_array_equal(
        global_assignments,
        np.asarray(expected["classificationGlobal"]),
    )
