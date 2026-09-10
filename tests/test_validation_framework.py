from pathlib import Path

from lunar_core.validation import build_validation_matrix, summarize_validation_evidence


def test_summarize_validation_evidence_reads_manifest():
    report = summarize_validation_evidence(Path("evidence/real_data_manifest.json"))

    assert report["validation_scope"] == "metadata_and_lazy_access_only"
    assert report["real_image_validation_status"] == "pending"
    assert report["capabilities"]["registration_executed"] is False
    assert "OHRC ↔ TMC-2" in report["validation_matrix"]


def test_build_validation_matrix_has_all_required_pairs():
    matrix = build_validation_matrix(Path("evidence/real_data_manifest.json"))

    required_pairs = {
        "OHRC ↔ TMC-2",
        "OHRC ↔ IIRS",
        "TMC-2 ↔ IIRS",
        "Chandrayaan-2 ↔ LRO NAC",
        "Chandrayaan-2 ↔ SELENE",
    }
    assert required_pairs.issubset(set(matrix.keys()))
    matrix_rows = matrix["OHRC ↔ TMC-2"]
    assert matrix_rows["ingestion"] in {"REAL DATA TESTED", "NOT AVAILABLE", "NOT RUN"}
    assert matrix_rows["ground_truth"] in {"NOT AVAILABLE", "NOT RUN", "N/A"}
