import csv
import json
from pathlib import Path
import pytest

from lunar_core.cli import cmd_catalog_show, cmd_evaluate, cmd_pair_discover, cmd_register


def _manifest(path: Path) -> None:
    fields = [
        "mission",
        "instrument",
        "product_id",
        "image_path",
        "gsd_m",
        "center_lat_deg",
        "center_lon_deg",
        "status",
    ]
    rows = [
        {
            "mission": "Chandrayaan-2",
            "instrument": "OHRC",
            "product_id": "source",
            "image_path": "/tmp/source.img",
            "gsd_m": "0.28",
            "center_lat_deg": "10.0",
            "center_lon_deg": "20.0",
            "status": "validated",
        },
        {
            "mission": "LRO",
            "instrument": "NAC",
            "product_id": "target",
            "image_path": "/tmp/target.img",
            "gsd_m": "0.5",
            "center_lat_deg": "10.2",
            "center_lon_deg": "20.2",
            "status": "validated",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_catalog_show_prints_manifest(tmp_path, capsys):
    manifest = tmp_path / "products.csv"
    _manifest(manifest)
    cmd_catalog_show(type("Args", (), {"manifest": str(manifest)})())
    assert json.loads(capsys.readouterr().out)[0]["product_id"] == "source"


def test_pair_discover_reports_proximity_candidate(tmp_path, capsys):
    manifest = tmp_path / "products.csv"
    _manifest(manifest)
    args = type(
        "Args",
        (),
        {"manifest": str(manifest), "mission": None, "candidates_only": True},
    )()
    cmd_pair_discover(args)
    result = json.loads(capsys.readouterr().out)
    assert result[0]["status"] == "proximity_candidate"
    assert result[0]["overlap_status"] == "UNKNOWN"


def test_evaluate_prints_existing_report(tmp_path, capsys):
    report = tmp_path / "evaluation.json"
    report.write_text('{"dataset_class": "synthetic"}', encoding="utf-8")
    cmd_evaluate(type("Args", (), {"report": str(report)})())
    assert "synthetic" in capsys.readouterr().out


def test_register_forwards_generic_source_and_target(monkeypatch):
    captured = {}

    def fake_run(command):
        captured["command"] = command
        return type("Result", (), {"returncode": 0})()

    import lunar_core.cli as cli_module
    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exit_info:
        cmd_register(
            type(
                "Args",
                (),
                {
                    "source": "source.img",
                    "target": "target.img",
                    "raw_dir": "raw",
                    "output_dir": "results",
                    "site": "test-site",
                },
            )()
        )

    assert exit_info.value.code == 0
    assert "--source" in captured["command"]
    assert "--target" in captured["command"]
    assert "--chandrayaan" not in captured["command"]
    assert "--lro" not in captured["command"]
