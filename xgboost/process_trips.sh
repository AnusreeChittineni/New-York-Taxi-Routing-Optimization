# NOTE: code generated with LLMs. Prompts used:
# Create a bash script to issue multiple slurm jobs in parallel with srun, using the extract_osrm_feats.py script


#!/usr/bin/env bash
set -euo pipefail

# ------------- CONFIG -------------
CHUNK_SIZE=100000                # data rows per chunk (excluding header)
PYTHON_BIN=python                # or python3
PY_SCRIPT="extract_osrm_feats.py" # path to your Python script
# ----------------------------------

usage() {
  cat <<EOF
Usage: $0 INPUT_CSV OSRM_URL [OUTPUT_DIR]

  INPUT_CSV   Path to the big input CSV.
  OSRM_URL    Base URL of your OSRM server (e.g. http://localhost:5000).
  OUTPUT_DIR  Directory to store chunked CSVs and their processed outputs.
              Default: <INPUT_BASENAME>_chunks next to INPUT_CSV.

The script will:
  - Split INPUT_CSV into ${CHUNK_SIZE}-row chunks (each with header).
  - Run OSRM processing on each chunk (srun parallel if available).
  - Concatenate all *_osrm.csv into one merged file.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

INPUT_CSV=$1
OSRM_URL=$2
OUTPUT_DIR=${3:-}

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "ERROR: Input CSV '$INPUT_CSV' not found" >&2
  exit 1
fi

# Derive default OUTPUT_DIR if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
  input_dir=$(dirname "$INPUT_CSV")
  input_base=$(basename "$INPUT_CSV")
  input_base_noext=${input_base%.*}
  OUTPUT_DIR="${input_dir}/${input_base_noext}_chunks"
fi

mkdir -p "$OUTPUT_DIR"

echo "Input CSV  : $INPUT_CSV"
echo "OSRM URL   : $OSRM_URL"
echo "Output dir : $OUTPUT_DIR"
echo "Chunk size : $CHUNK_SIZE"

# ---- Split CSV into chunks of CHUNK_SIZE (data rows), keeping header ----

header_file="${OUTPUT_DIR}/header.tmp"
data_file="${OUTPUT_DIR}/data.tmp"

# Extract header and data
head -n 1 "$INPUT_CSV" > "$header_file"
tail -n +2 "$INPUT_CSV" > "$data_file"

echo "Splitting data into chunks of ${CHUNK_SIZE} rows..."

# Use split to create chunk files without header
split -d -l "$CHUNK_SIZE" "$data_file" "${OUTPUT_DIR}/chunk_"

rm -f "$data_file"

# Prepend header to each chunk and rename to .csv
for f in "${OUTPUT_DIR}"/chunk_[0-9][0-9]*; do
  [[ "$f" == *.csv ]] && continue
  tmp="${f}.tmp"
  mv "$f" "$tmp"
  {
    printf '%s\n' "$(cat "$header_file")"
    cat "$tmp"
  } > "${f}.csv"
  rm -f "$tmp"
done

rm -f "$header_file"

chunks=( "${OUTPUT_DIR}"/chunk_*.csv )

if [[ ${#chunks[@]} -eq 0 ]]; then
  echo "No chunks created; nothing to do."
  exit 0
fi

echo "Created ${#chunks[@]} chunk file(s)."

# ---- Decide whether to use srun or plain background jobs ----

use_srun=false
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  use_srun=true
  echo "Detected Slurm environment; will use srun for parallel jobs."
else
  echo "No Slurm job detected (or srun not found); will use background jobs."
fi

# ---- Run Python script for each chunk in parallel ----

for chunk in "${chunks[@]}"; do
  out_csv="${chunk%.csv}_osrm.csv"

  echo "Scheduling: $chunk -> $out_csv"

  if $use_srun; then
    # Each chunk as its own Slurm task
    srun --exclusive -N1 -n1 \
      "$PYTHON_BIN" "$PY_SCRIPT" \
        --input_csv "$chunk" \
        --output_csv "$out_csv" \
        --osrm_server_base_url "$OSRM_URL" &
  else
    # Plain background job
    "$PYTHON_BIN" "$PY_SCRIPT" \
      --input_csv "$chunk" \
      --output_csv "$out_csv" \
      --osrm_server_base_url "$OSRM_URL" &
  fi
done

echo "Waiting for all chunks to finish..."
wait
echo "All chunk OSRM processing complete."

# ---- CONCATENATE ALL *_osrm.csv INTO A SINGLE MERGED FILE ----

echo "Concatenating chunk outputs..."

merged="${OUTPUT_DIR}/merged_osrm.csv"

# Sorted filenames ensure correct ordering: chunk_00, chunk_01, ...
osrm_chunks=( $(ls "${OUTPUT_DIR}"/chunk_*_osrm.csv | sort) )

if [[ ${#osrm_chunks[@]} -eq 0 ]]; then
  echo "ERROR: No *_osrm.csv files found. Nothing to merge." >&2
  exit 1
fi

# Write header from first file
head -n 1 "${osrm_chunks[0]}" > "$merged"

# Append all bodies
for f in "${osrm_chunks[@]}"; do
  tail -n +2 "$f" >> "$merged"
done

echo "Merged file created at:"
echo "  $merged"

echo "Done."
