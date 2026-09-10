import csv
import json
import subprocess
import sys
from pathlib import Path
import pytest

from lunar_core.cli import cmd_catalog_show, cmd_evaluate, cmd_pair_discover, cmd_register
from samanvaya.data_io.import_real_pair import import_real_pair


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


def test_discover_data_cli_reports_ohrc_verified_and_lroc_missing(tmp_path):
    source = tmp_path / "CH2_OHRC_20240601_0001.img"
    source.write_bytes(b"\x00" * 64)
    (tmp_path / "CH2_OHRC_20240601_0001.xml").write_text(
        """
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
            <Observation_Area>
                <Time_Coordinates><start_date_time>2024-06-01T00:00:00Z</start_date_time></Time_Coordinates>
                <Mission_Area>
                    <Product_Parameters><pixel_resolution unit="m/pixel">0.28</pixel_resolution></Product_Parameters>
                </Mission_Area>
            </Observation_Area>
            <File_Area_Observational>
                <Array_2D_Image>
                    <Axis_Array><elements>64</elements></Axis_Array>
                    <Axis_Array><elements>64</elements></Axis_Array>
                    <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
                </Array_2D_Image>
            </File_Area_Observational>
        </Product_Observational>
        """,
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "samanvaya", "discover-data", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0
    assert "OHRC" in result.stdout
    assert "VERIFIED" in result.stdout
    assert "LROC" in result.stdout
    assert "NOT FOUND" in result.stdout
    assert "No verified real pair available" in result.stdout


def test_import_real_pair_reports_missing_lroc_blocker(tmp_path):
    source = tmp_path / "CH2_OHRC_20240601_0001.img"
    source.write_bytes(b"\x00" * 64)
    (tmp_path / "CH2_OHRC_20240601_0001.xml").write_text(
        """
        <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
            <Observation_Area>
                <Time_Coordinates><start_date_time>2024-06-01T00:00:00Z</start_date_time></Time_Coordinates>
                <Mission_Area>
                    <Product_Parameters><pixel_resolution unit="m/pixel">0.28</pixel_resolution></Product_Parameters>
                </Mission_Area>
            </Observation_Area>
            <File_Area_Observational>
                <Array_2D_Image>
                    <Axis_Array><elements>64</elements></Axis_Array>
                    <Axis_Array><elements>64</elements></Axis_Array>
                    <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
                </Array_2D_Image>
            </File_Area_Observational>
        </Product_Observational>
        """,
        encoding="utf-8",
    )

    missing_lroc = tmp_path / "missing_nac.img"
    with pytest.raises(FileNotFoundError, match="LROC NAC product not found"):
        import_real_pair(source, missing_lroc, manifest_path=tmp_path / "real_manifest.json")
