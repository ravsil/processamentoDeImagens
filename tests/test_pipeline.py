"""Integration tests for the end-to-end PathoShift pipeline."""

from pathlib import Path
import json
import pytest
from typer.testing import CliRunner

from pathoshift.cli import app
from pathoshift.data.synthetic import create_synthetic_dataset

runner = CliRunner()


def test_full_pipeline_cli_flow(tmp_path: Path):
    data_dir = tmp_path / "raw_data"
    create_synthetic_dataset(data_dir, n_patients=4, tiles_per_patient=4)

    output_dir = tmp_path / "outputs"

    # 1. Prepare
    res_prep = runner.invoke(app, ["prepare", "--input", str(data_dir), "--output", str(output_dir)])
    assert res_prep.exit_code == 0
    manifest_path = output_dir / "manifest.jsonl"
    assert manifest_path.exists()

    # 2. EDA
    eda_dir = output_dir / "eda"
    res_eda = runner.invoke(app, ["eda", "--manifest", str(manifest_path), "--output", str(eda_dir)])
    assert res_eda.exit_code == 0
    assert (eda_dir / "eda_summary.json").exists()

    # 3. Describe
    res_desc = runner.invoke(app, ["describe"])
    assert res_desc.exit_code == 0
