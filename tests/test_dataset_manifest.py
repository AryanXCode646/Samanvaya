from pathlib import Path

from lunar_core.data_io.dataset_manifest import load_dataset_manifest, summarize_dataset_status


def test_dataset_manifest_loads_and_validates():
    manifest = load_dataset_manifest(Path("datasets/manifest.yaml"))
    assert "datasets" in manifest
    assert len(manifest["datasets"]) >= 5
    assert any(dataset["instrument"] == "OHRC" for dataset in manifest["datasets"])


def test_dataset_manifest_summary_is_conservative():
    summary = summarize_dataset_status(Path("datasets/manifest.yaml"))
    assert summary["total_datasets"] >= 5
    assert "pending_real_data" in summary
    assert summary["by_status"].get("pending", 0) >= 1
