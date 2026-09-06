#!/usr/bin/env bash
# Stage official Home Credit Parquet files from Kaggle into the SageMaker S3 bucket.
# Files are staged on persistent project storage, hashed, encrypted, uploaded,
# verified, and then removed from staging. The durable raw snapshot stays in S3.

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

COMPETITION="home-credit-credit-risk-model-stability"
STAGING_ROOT="${HOME_CREDIT_STAGING_ROOT:-$PROJECT_ROOT/data/staging}"
MIN_STAGING_FREE_GIB="${MIN_STAGING_FREE_GIB:-30}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
START_EPOCH="$(date +%s)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="logs/kaggle-to-s3-${STAMP}.log"
INVENTORY_CSV="artifacts/kaggle-files-${STAMP}.csv"
SELECTED_FILES="artifacts/kaggle-selected-files-${STAMP}.txt"
MANIFEST_JSONL="artifacts/kaggle-to-s3-${STAMP}.jsonl"

mkdir -p logs artifacts "$STAGING_ROOT"

stamp() { date -u +"[%Y-%m-%dT%H:%M:%SZ]"; }
log() { echo "$(stamp) $*" | tee -a "$LOG_FILE"; }

run_with_heartbeat() {
    local label="$1"
    shift
    local started pid rc elapsed next_heartbeat
    started="$(date +%s)"
    next_heartbeat="$HEARTBEAT_SECONDS"
    log "stage_started stage=$label"
    "$@" >>"$LOG_FILE" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
        elapsed=$(( $(date +%s) - started ))
        if kill -0 "$pid" 2>/dev/null && [ "$elapsed" -ge "$next_heartbeat" ]; then
            log "heartbeat stage=$label elapsed_seconds=$elapsed pid=$pid"
            next_heartbeat=$(( next_heartbeat + HEARTBEAT_SECONDS ))
        fi
    done
    wait "$pid"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    if [ "$rc" -ne 0 ]; then
        log "stage_failed stage=$label exit_code=$rc elapsed_seconds=$elapsed"
        return "$rc"
    fi
    log "stage_completed stage=$label elapsed_seconds=$elapsed"
    return 0
}

resolve_region() {
    if [ -n "${AWS_REGION:-}" ]; then
        printf '%s\n' "$AWS_REGION"
        return
    fi
    if [ -n "${AWS_DEFAULT_REGION:-}" ]; then
        printf '%s\n' "$AWS_DEFAULT_REGION"
        return
    fi
    aws configure get region 2>/dev/null || true
}

create_bucket_if_needed() {
    local bucket="$1"
    local region="$2"
    if aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
        log "s3_bucket_ready bucket=$bucket created=false"
        return 0
    fi

    log "s3_bucket_create_started bucket=$bucket region=$region"
    if [ "$region" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$bucket" >/dev/null || return $?
    else
        aws s3api create-bucket \
            --bucket "$bucket" \
            --create-bucket-configuration "LocationConstraint=$region" \
            >/dev/null || return $?
    fi
    log "s3_bucket_ready bucket=$bucket created=true"
}

verify_s3_write_access() {
    local bucket="$1"
    local prefix="$2"
    local probe_file probe_key remote_sse
    probe_file="$STAGING_ROOT/s3-write-probe.txt"
    probe_key="$prefix/_probes/write-probe-${STAMP}.txt"
    printf 'home-credit-s3-write-probe %s\n' "$STAMP" >"$probe_file"

    aws s3 cp \
        "$probe_file" \
        "s3://$bucket/$probe_key" \
        --sse AES256 \
        --only-show-errors >/dev/null || return $?

    remote_sse="$(aws s3api head-object \
        --bucket "$bucket" \
        --key "$probe_key" \
        --query ServerSideEncryption \
        --output text)" || return $?
    if [ "$remote_sse" != "AES256" ]; then
        log "s3_probe_failed reason=encryption_mismatch server_side_encryption=$remote_sse"
        return 1
    fi

    aws s3 rm "s3://$bucket/$probe_key" --only-show-errors >/dev/null || return $?
    rm -f "$probe_file"
    log "s3_write_probe_passed bucket=$bucket prefix=$prefix encryption=AES256"
}

select_kaggle_files() {
    python3 - "$INVENTORY_CSV" "$SELECTED_FILES" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

with source.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if not rows or "name" not in rows[0]:
    raise SystemExit("Kaggle file inventory is empty or missing the 'name' column")

selected = sorted(
    row["name"]
    for row in rows
    if row["name"].endswith(".parquet")
    or Path(row["name"]).name == "sample_submission.csv"
)

if not selected:
    raise SystemExit("No Parquet files were found in the Kaggle competition inventory")

destination.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"selected_files={len(selected)}")
PY
}

prepare_downloaded_file() {
    local work_dir="$1"
    local expected_name="$2"
    local expected_basename
    expected_basename="$(basename "$expected_name")"

    local candidate
    candidate="$(find "$work_dir" -type f -name "$expected_basename" -print -quit)"
    if [ -n "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    local archive
    archive="$(find "$work_dir" -maxdepth 1 -type f -name '*.zip' -print -quit)"
    if [ -z "$archive" ]; then
        return 1
    fi

    python3 -m zipfile -e "$archive" "$work_dir/extracted" || return $?
    candidate="$(find "$work_dir/extracted" -type f -name "$expected_basename" -print -quit)"
    if [ -z "$candidate" ]; then
        return 1
    fi
    printf '%s\n' "$candidate"
}

log "kaggle_to_s3_started staging_root=$STAGING_ROOT"

if [ ! -f uv.lock ]; then
    log "kaggle_to_s3_failed reason=uv_lock_missing"
    exit 1
fi

uv run --locked home-credit data-preflight \
    --path "$STAGING_ROOT" \
    --min-free-gib "$MIN_STAGING_FREE_GIB" >>"$LOG_FILE" 2>&1 || {
    rc=$?
    log "kaggle_to_s3_failed reason=staging_capacity exit_code=$rc"
    exit "$rc"
}

# Verify Kaggle access before creating or probing any AWS storage resources.
log "kaggle_inventory_started competition=$COMPETITION"
uv run --locked kaggle competitions files \
    "$COMPETITION" \
    --page-size 200 \
    -v -q >"$INVENTORY_CSV" || {
    rc=$?
    log "kaggle_to_s3_failed reason=kaggle_access exit_code=$rc"
    log "next_action=uv run --locked kaggle auth login"
    exit "$rc"
}
select_kaggle_files | tee -a "$LOG_FILE" || exit $?
TOTAL_FILES="$(wc -l < "$SELECTED_FILES" | tr -d ' ')"
log "kaggle_inventory_completed selected_files=$TOTAL_FILES"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)" || exit $?
REGION="$(resolve_region)"
if [ -z "$REGION" ]; then
    log "kaggle_to_s3_failed reason=aws_region_missing"
    exit 1
fi

# SageMaker execution roles commonly have object permissions on buckets whose
# names contain "sagemaker". Use the conventional SageMaker default bucket name
# unless the caller deliberately supplies an accessible bucket.
BUCKET="${HOME_CREDIT_S3_BUCKET:-sagemaker-${REGION}-${ACCOUNT_ID}}"
PREFIX="${HOME_CREDIT_S3_PREFIX:-home-credit-model-stability/raw/kaggle}"

create_bucket_if_needed "$BUCKET" "$REGION" || {
    rc=$?
    log "kaggle_to_s3_failed reason=s3_bucket_access exit_code=$rc bucket=$BUCKET"
    exit "$rc"
}
verify_s3_write_access "$BUCKET" "$PREFIX" || {
    rc=$?
    log "kaggle_to_s3_failed reason=s3_write_access exit_code=$rc bucket=$BUCKET"
    exit "$rc"
}

INDEX=0
while IFS= read -r FILE_NAME; do
    [ -n "$FILE_NAME" ] || continue
    INDEX=$((INDEX + 1))
    S3_KEY="$PREFIX/$FILE_NAME"

    if aws s3api head-object \
        --bucket "$BUCKET" \
        --key "$S3_KEY" >/dev/null 2>&1; then
        log "file_skipped index=$INDEX total=$TOTAL_FILES file=$FILE_NAME reason=s3_exists"
        continue
    fi

    WORK_DIR="$STAGING_ROOT/current"
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    log "file_started index=$INDEX total=$TOTAL_FILES file=$FILE_NAME"

    run_with_heartbeat "kaggle_download_${INDEX}" \
        uv run --locked kaggle competitions download \
        "$COMPETITION" \
        -f "$FILE_NAME" \
        -p "$WORK_DIR" \
        -o -q || exit $?

    LOCAL_FILE="$(prepare_downloaded_file "$WORK_DIR" "$FILE_NAME")" || {
        log "file_failed file=$FILE_NAME reason=downloaded_payload_not_found"
        exit 1
    }

    BYTES="$(stat -c '%s' "$LOCAL_FILE")"
    SHA256="$(sha256sum "$LOCAL_FILE" | awk '{print $1}')"

    run_with_heartbeat "s3_upload_${INDEX}" \
        aws s3 cp \
        "$LOCAL_FILE" \
        "s3://$BUCKET/$S3_KEY" \
        --sse AES256 \
        --metadata "sha256=$SHA256" \
        --only-show-errors || exit $?

    read -r REMOTE_BYTES REMOTE_SSE REMOTE_SHA256 <<<"$(aws s3api head-object \
        --bucket "$BUCKET" \
        --key "$S3_KEY" \
        --query '[ContentLength,ServerSideEncryption,Metadata.sha256]' \
        --output text)" || exit $?

    if [ "$REMOTE_BYTES" != "$BYTES" ]; then
        log "file_failed file=$FILE_NAME reason=size_mismatch local=$BYTES remote=$REMOTE_BYTES"
        exit 1
    fi
    if [ "$REMOTE_SSE" != "AES256" ]; then
        log "file_failed file=$FILE_NAME reason=encryption_mismatch remote_sse=$REMOTE_SSE"
        exit 1
    fi
    if [ "$REMOTE_SHA256" != "$SHA256" ]; then
        log "file_failed file=$FILE_NAME reason=sha256_metadata_mismatch"
        exit 1
    fi

    python3 - "$MANIFEST_JSONL" "$FILE_NAME" "$S3_KEY" "$BYTES" "$SHA256" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

manifest, name, key, size, digest = sys.argv[1:]
record = {
    "file": name,
    "s3_key": key,
    "bytes": int(size),
    "sha256": digest,
    "staged_at_utc": datetime.now(UTC).isoformat(),
}
with Path(manifest).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY

    rm -rf "$WORK_DIR"
    log "file_completed index=$INDEX total=$TOTAL_FILES file=$FILE_NAME bytes=$BYTES sha256=$SHA256"
done < "$SELECTED_FILES"

MANIFEST_KEY="home-credit-model-stability/manifests/$(basename "$MANIFEST_JSONL")"
INVENTORY_KEY="home-credit-model-stability/manifests/$(basename "$INVENTORY_CSV")"
aws s3 cp \
    "$MANIFEST_JSONL" \
    "s3://$BUCKET/$MANIFEST_KEY" \
    --sse AES256 \
    --only-show-errors || exit $?
aws s3 cp \
    "$INVENTORY_CSV" \
    "s3://$BUCKET/$INVENTORY_KEY" \
    --sse AES256 \
    --only-show-errors || exit $?

TOTAL_SECONDS=$(( $(date +%s) - START_EPOCH ))
log "kaggle_to_s3_completed bucket=$BUCKET prefix=$PREFIX total_seconds=$TOTAL_SECONDS"
log "manifest_s3_uri=s3://$BUCKET/$MANIFEST_KEY"
exit 0
