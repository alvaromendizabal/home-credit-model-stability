from __future__ import annotations

from pathlib import Path


def test_data_acquisition_is_s3_first_with_persistent_staging() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "download_data.sh").read_text(encoding="utf-8")

    assert "$PROJECT_ROOT/data/staging" in script
    assert "aws s3 cp" in script
    assert 'row["name"].endswith(".parquet")' in script
    assert 'rm -rf "$WORK_DIR"' in script
    assert "HEARTBEAT_SECONDS" in script
    assert "--sse AES256" in script
    assert '--metadata "sha256=$SHA256"' in script


def test_data_acquisition_uses_sagemaker_bucket_without_bucket_admin_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "download_data.sh").read_text(encoding="utf-8")

    assert "sagemaker-${REGION}-${ACCOUNT_ID}" in script
    assert "put-public-access-block" not in script
    assert "put-bucket-encryption" not in script
    assert "put-bucket-versioning" not in script
    assert "verify_s3_write_access" in script


def test_kaggle_access_is_verified_before_s3_creation() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "download_data.sh").read_text(encoding="utf-8")

    inventory_position = script.index('log "kaggle_inventory_started')
    bucket_position = script.index('create_bucket_if_needed "$BUCKET"')
    assert inventory_position < bucket_position
    assert "uv run --locked kaggle auth login" in script


def test_data_acquisition_uses_project_staging_without_full_local_dataset() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "download_data.sh").read_text(encoding="utf-8")

    assert 'STAGING_ROOT="${HOME_CREDIT_STAGING_ROOT:-$PROJECT_ROOT/data/staging}"' in script
    assert 'DEST="data/downloads"' not in script
    assert 'MIN_FREE_GIB="${MIN_FREE_GIB:-60}"' not in script
