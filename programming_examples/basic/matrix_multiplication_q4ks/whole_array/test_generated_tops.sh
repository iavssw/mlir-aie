#!/usr/bin/env bash

# Benchmark already-generated Q4_K whole-array xclbin/instruction pairs.
# Artifact filenames are the source of truth for dimensions and build modes.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_SCRIPT="/home/greg/Desktop/env-mlir-aie.sh"
cd "$SCRIPT_DIR"
set -e

BUILD_DIR="$SCRIPT_DIR/build"
RUNNER="$SCRIPT_DIR/whole_array_q4ks.exe"
WARMUP=4
ITERS=10
ROUNDS=1
VERIFY=true
VERIFY_SAMPLES=1000
MIN_TOPS=0
TIMEOUT_SECONDS=900
OUTPUT=""
LIST_ONLY=false
DRY_RUN=false
M_MIN=0
M_MAX=0
declare -a SHAPE_FILTERS=()
declare -a M_FILTERS=()

usage() {
  cat <<'EOF'
Usage: ./test_generated_tops.sh [options]

Discover matching final_*.xclbin and insts_*.bin files in build/, execute the
Q4_K host runner, and report effective TOPS = 2*M*K*N / NPU_time / 1e12.

Options:
  --shape KxN          Test only this K-by-N family (repeatable).
  --m M                Test only this M value (repeatable).
  --m-min M            Skip artifacts with smaller M.
  --m-max M            Skip artifacts with larger M.
  --warmup N           Warmup iterations per round (default: 4).
  --iters N            Timed iterations per round (default: 10).
  --rounds N           Independent host runs (default: 1).
  --verify             Enable sampled/full verification (default).
  --no-verify          Throughput-only diagnostic; marked in CSV.
  --verify-samples N   Samples for large matrices (default: 1000).
  --min-tops VALUE     Fail if a configuration's median is below VALUE.
  --timeout SECONDS    Per-round timeout (default: 900).
  --build-dir PATH     Artifact directory (default: ./build).
  --runner PATH        C++ host executable (default: ./whole_array_q4ks.exe).
  --output PATH        CSV path (default: build/benchmark_generated_TIMESTAMP.csv).
  --list               List matching pairs without touching the NPU.
  --dry-run            Print commands without touching the NPU.
  -h, --help           Show this help.

Examples:
  ./test_generated_tops.sh --list
  ./test_generated_tops.sh --shape 4096x14336 --m-min 3072 --m-max 4096
  ./test_generated_tops.sh --shape 14336x4096 --m 4096 \
      --warmup 16 --iters 20 --rounds 3 --min-tops 12

The script stops after the first failed or timed-out kernel. Do not rerun it
after an XRT/kernel failure until the NPU driver is healthy (normally a reboot).
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

require_value() {
  (($# >= 2)) || die "$1 requires a value"
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

while (($#)); do
  case "$1" in
    --shape)
      require_value "$@"
      [[ "$2" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]] ||
        die "--shape must be KxN, for example 4096x14336"
      SHAPE_FILTERS+=("$2")
      shift 2
      ;;
    --m)
      require_value "$@"
      is_positive_integer "$2" || die "--m must be a positive integer"
      M_FILTERS+=("$2")
      shift 2
      ;;
    --m-min)
      require_value "$@"
      is_nonnegative_integer "$2" || die "--m-min must be non-negative"
      M_MIN="$2"
      shift 2
      ;;
    --m-max)
      require_value "$@"
      is_nonnegative_integer "$2" || die "--m-max must be non-negative"
      M_MAX="$2"
      shift 2
      ;;
    --warmup)
      require_value "$@"
      is_nonnegative_integer "$2" || die "--warmup must be non-negative"
      WARMUP="$2"
      shift 2
      ;;
    --iters)
      require_value "$@"
      is_positive_integer "$2" || die "--iters must be positive"
      ITERS="$2"
      shift 2
      ;;
    --rounds)
      require_value "$@"
      is_positive_integer "$2" || die "--rounds must be positive"
      ROUNDS="$2"
      shift 2
      ;;
    --verify)
      VERIFY=true
      shift
      ;;
    --no-verify)
      VERIFY=false
      shift
      ;;
    --verify-samples)
      require_value "$@"
      is_positive_integer "$2" || die "--verify-samples must be positive"
      VERIFY_SAMPLES="$2"
      shift 2
      ;;
    --min-tops)
      require_value "$@"
      is_nonnegative_number "$2" || die "--min-tops must be non-negative"
      MIN_TOPS="$2"
      shift 2
      ;;
    --timeout)
      require_value "$@"
      is_positive_integer "$2" || die "--timeout must be positive"
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --build-dir)
      require_value "$@"
      BUILD_DIR="$2"
      shift 2
      ;;
    --runner)
      require_value "$@"
      RUNNER="$2"
      shift 2
      ;;
    --output)
      require_value "$@"
      OUTPUT="$2"
      shift 2
      ;;
    --list)
      LIST_ONLY=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -d "$BUILD_DIR" ]] || die "build directory does not exist: $BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

selected_shape() {
  local candidate="$1"
  local requested
  ((${#SHAPE_FILTERS[@]} == 0)) && return 0
  for requested in "${SHAPE_FILTERS[@]}"; do
    [[ "$candidate" == "$requested" ]] && return 0
  done
  return 1
}

selected_m() {
  local candidate="$1"
  local requested
  ((M_MIN == 0 || candidate >= M_MIN)) || return 1
  ((M_MAX == 0 || candidate <= M_MAX)) || return 1
  ((${#M_FILTERS[@]} == 0)) && return 0
  for requested in "${M_FILTERS[@]}"; do
    [[ "$candidate" == "$requested" ]] && return 0
  done
  return 1
}

# Records are sorted by K, N, M. Generated artifact names contain no pipes.
declare -a ARTIFACTS=()
shopt -s nullglob
for xclbin in "$BUILD_DIR"/final_*.xclbin; do
  base="${xclbin##*/}"
  if [[ "$base" =~ ^final_([0-9]+)x([0-9]+)x([0-9]+)_([0-9]+)a([0-9]+)x([0-9]+)x([0-9]+)_([0-9]+)c_([^_]+)_([^_]+)_(.+)[.]xclbin$ ]]; then
    M="${BASH_REMATCH[1]}"
    K="${BASH_REMATCH[2]}"
    N="${BASH_REMATCH[3]}"
    M_C="${BASH_REMATCH[4]}"
    M_A="${BASH_REMATCH[5]}"
    K_TILE="${BASH_REMATCH[6]}"
    N_TILE="${BASH_REMATCH[7]}"
    COLS="${BASH_REMATCH[8]}"
    COMPUTE_TYPE="${BASH_REMATCH[9]}"
    ACCUMULATION_MODE="${BASH_REMATCH[10]}"
    CACHE_MODE="${BASH_REMATCH[11]}"
    selected_shape "${K}x${N}" || continue
    selected_m "$M" || continue
    artifact_stem="${base#final_}"
    artifact_stem="${artifact_stem%.xclbin}"
    insts="$BUILD_DIR/insts_${artifact_stem}.bin"
    [[ -f "$insts" ]] || {
      echo "warning: skipping $base because ${insts##*/} is missing" >&2
      continue
    }
    ARTIFACTS+=("$M|$K|$N|$M_C|$M_A|$K_TILE|$N_TILE|$COLS|$COMPUTE_TYPE|$ACCUMULATION_MODE|$CACHE_MODE|$xclbin|$insts")
  fi
done
shopt -u nullglob

((${#ARTIFACTS[@]} > 0)) || die "no complete generated kernel pairs matched"
mapfile -t ARTIFACTS < <(printf '%s\n' "${ARTIFACTS[@]}" | sort -t '|' -k2,2n -k3,3n -k1,1n)

print_artifacts() {
  printf '%-7s %-7s %-7s %-16s %-5s %-9s %-15s %-16s %s\n' \
    M K N TILE COLS COMPUTE ACCUMULATION CACHE XCLBIN
  local record M K N M_C M_A K_TILE N_TILE COLS COMPUTE_TYPE
  local ACCUMULATION_MODE CACHE_MODE xclbin insts
  for record in "${ARTIFACTS[@]}"; do
    IFS='|' read -r M K N M_C M_A K_TILE N_TILE COLS COMPUTE_TYPE \
      ACCUMULATION_MODE CACHE_MODE xclbin insts <<< "$record"
    printf '%-7s %-7s %-7s %-16s %-5s %-9s %-15s %-16s %s\n' \
      "$M" "$K" "$N" "${M_C}/${M_A}x${K_TILE}x${N_TILE}" "$COLS" \
      "$COMPUTE_TYPE" "$ACCUMULATION_MODE" "$CACHE_MODE" "${xclbin##*/}"
  done
  echo "Matched ${#ARTIFACTS[@]} complete artifact pair(s)."
}

if [[ "$LIST_ONLY" == true ]]; then
  print_artifacts
  exit 0
fi

if [[ "$DRY_RUN" == false ]]; then
  [[ -f "$ENV_SCRIPT" ]] || die "environment script does not exist: $ENV_SCRIPT"
  # env-mlir-aie.sh uses relative paths and currently ends with an obsolete cd.
  # Source it from Desktop after all CLI arguments have been consumed, tolerate
  # only that final cd failure, then explicitly return to this directory.
  cd /home/greg/Desktop
  # shellcheck disable=SC1090
  source "$ENV_SCRIPT" || true
  cd "$SCRIPT_DIR"
  command -v timeout >/dev/null || die "timeout command is required"
  command -v xrt-smi >/dev/null || die "xrt-smi is unavailable after sourcing $ENV_SCRIPT"
  if [[ ! -x "$RUNNER" ]]; then
    echo "Building host runner only (no xclbin will be rebuilt)..."
    make -C "$SCRIPT_DIR" whole_array_q4ks.exe
  fi
  [[ -x "$RUNNER" ]] || die "host runner is not executable: $RUNNER"
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$BUILD_DIR/benchmark_generated_$(date +%Y%m%d_%H%M%S).csv"
elif [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$SCRIPT_DIR/$OUTPUT"
fi

if [[ "$DRY_RUN" == false ]]; then
  [[ ! -e "$OUTPUT" ]] || die "refusing to overwrite existing output: $OUTPUT"
  mkdir -p "$(dirname "$OUTPUT")"
  LOG_DIR="${OUTPUT%.csv}_logs"
  mkdir -p "$LOG_DIR"
  printf '%s\n' 'record_type,timestamp,M,K,N,tile_m_c,tile_m_a,tile_k,tile_n,columns,compute_type,accumulation_mode,cache_mode,round,warmup,iters,verified,avg_us,gflops,tops,xclbin,insts,log' > "$OUTPUT"

  PREFLIGHT_LOG="$LOG_DIR/xrt_preflight.txt"
  echo "Checking XRT/NPU responsiveness before loading the first kernel..."
  set +e
  timeout --signal=INT --kill-after=5s 20s xrt-smi examine > "$PREFLIGHT_LOG" 2>&1
  PREFLIGHT_STATUS=$?
  set -e
  if ((PREFLIGHT_STATUS != 0)); then
    echo "NPU/XRT preflight failed or timed out (status $PREFLIGHT_STATUS)." >&2
    echo "No generated kernel was loaded. Reboot before benchmarking." >&2
    echo "Preflight log: $PREFLIGHT_LOG" >&2
    exit 1
  fi
fi

csv_quote() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

median_of() {
  printf '%s\n' "$@" | sort -g | awk '
    { values[NR] = $1 }
    END {
      if (NR % 2) printf "%.9f", values[(NR + 1) / 2];
      else printf "%.9f", (values[NR / 2] + values[NR / 2 + 1]) / 2.0;
    }'
}

append_csv() {
  local record_type="$1" round="$2" avg_us="$3" gflops="$4" tops="$5"
  local timestamp
  timestamp="$(date --iso-8601=seconds)"
  printf '%s,' "$record_type" >> "$OUTPUT"
  csv_quote "$timestamp" >> "$OUTPUT"
  printf ',%s,%s,%s,%s,%s,%s,%s,%s,' \
    "$M" "$K" "$N" "$M_C" "$M_A" "$K_TILE" "$N_TILE" "$COLS" >> "$OUTPUT"
  csv_quote "$COMPUTE_TYPE" >> "$OUTPUT"
  printf ',' >> "$OUTPUT"
  csv_quote "$ACCUMULATION_MODE" >> "$OUTPUT"
  printf ',' >> "$OUTPUT"
  csv_quote "$CACHE_MODE" >> "$OUTPUT"
  printf ',%s,%s,%s,%s,%s,%s,%s,' \
    "$round" "$WARMUP" "$ITERS" "$VERIFY" "$avg_us" "$gflops" "$tops" >> "$OUTPUT"
  csv_quote "$xclbin" >> "$OUTPUT"
  printf ',' >> "$OUTPUT"
  csv_quote "$insts" >> "$OUTPUT"
  printf ',' >> "$OUTPUT"
  csv_quote "$RUN_LOG" >> "$OUTPUT"
  printf '\n' >> "$OUTPUT"
}

diagnose_failure() {
  local status="$1" log="$2"
  echo >&2
  echo "STOPPED after failure for ${M}x${K}x${N}; no later xclbin was loaded." >&2
  echo "Exit status: $status" >&2
  echo "Log: $log" >&2
  if ((status == 124 || status == 137)) ||
     grep -Eiq 'kernel failed|ert_cmd|amdxdna|xrt.*(error|failed)|device.*(busy|error)|failed.*(device|context|xclbin)|timed out' "$log"; then
    echo "This looks like an NPU/XRT/kernel failure. Treat the driver as potentially wedged and reboot before retrying." >&2
  else
    echo "This appears to be a host validation or correctness failure, not proof that the driver is wedged." >&2
  fi
}

echo "Matched ${#ARTIFACTS[@]} generated kernel pair(s)."
echo "Settings: warmup=$WARMUP iters=$ITERS rounds=$ROUNDS verify=$VERIFY samples=$VERIFY_SAMPLES"
[[ "$VERIFY" == true ]] || echo "WARNING: verification is disabled; results are throughput-only."

for record in "${ARTIFACTS[@]}"; do
  IFS='|' read -r M K N M_C M_A K_TILE N_TILE COLS COMPUTE_TYPE \
    ACCUMULATION_MODE CACHE_MODE xclbin insts <<< "$record"

  cmd=(
    "$RUNNER"
    -x "$xclbin"
    -i "$insts"
    --kernel MLIR_AIE
    -M "$M"
    -K "$K"
    -N "$N"
    --warmup "$WARMUP"
    --iters "$ITERS"
    "--verify=$VERIFY"
    --verify-samples "$VERIFY_SAMPLES"
    --min-gflops 0
    --verbosity 0
    --b_col_maj 0
    --c_col_maj 0
    --tile-m-c "$M_C"
    --tile-m-a "$M_A"
    --tile-k "$K_TILE"
    --tile-n "$N_TILE"
    --n-aie-cols "$COLS"
    --compute-type "$COMPUTE_TYPE"
    --accumulation-mode "$ACCUMULATION_MODE"
    --cache-mode "$CACHE_MODE"
    --cache-k "$K"
  )

  if [[ "$DRY_RUN" == true ]]; then
    printf 'timeout %ss ' "$TIMEOUT_SECONDS"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi

  echo
  echo "Benchmarking M=$M K=$K N=$N, tile=${M_C}/${M_A}x${K_TILE}x${N_TILE}, $COMPUTE_TYPE/$ACCUMULATION_MODE/$CACHE_MODE"
  declare -a ROUND_US=()
  declare -a ROUND_GFLOPS=()
  declare -a ROUND_TOPS=()

  for ((round = 1; round <= ROUNDS; ++round)); do
    RUN_LOG="$LOG_DIR/${M}x${K}x${N}_${M_C}a${M_A}x${K_TILE}x${N_TILE}_${COLS}c_${COMPUTE_TYPE}_${ACCUMULATION_MODE}_${CACHE_MODE}_round${round}.log"
    echo "Round $round/$ROUNDS"
    set +e
    timeout --signal=INT --kill-after=15s "${TIMEOUT_SECONDS}s" "${cmd[@]}" 2>&1 | tee "$RUN_LOG"
    RUN_STATUS=${PIPESTATUS[0]}
    set -e
    if ((RUN_STATUS != 0)); then
      diagnose_failure "$RUN_STATUS" "$RUN_LOG"
      exit 1
    fi

    AVG_US="$(awk '/^Avg NPU matmul time:/ { value=$5; sub(/us[.]?$/, "", value); print value }' "$RUN_LOG" | tail -n 1)"
    GFLOPS="$(awk '/^Avg NPU gflops:/ { print $4 }' "$RUN_LOG" | tail -n 1)"
    [[ -n "$AVG_US" && -n "$GFLOPS" ]] || {
      echo "Could not parse timing output from $RUN_LOG" >&2
      exit 1
    }
    TOPS="$(awk -v value="$GFLOPS" 'BEGIN { printf "%.9f", value / 1000.0 }')"
    ROUND_US+=("$AVG_US")
    ROUND_GFLOPS+=("$GFLOPS")
    ROUND_TOPS+=("$TOPS")
    append_csv round "$round" "$AVG_US" "$GFLOPS" "$TOPS"
    printf 'Result: %.3f TOPS (%.3f GFLOP/s, %s us average)\n' "$TOPS" "$GFLOPS" "$AVG_US"
  done

  MEDIAN_US="$(median_of "${ROUND_US[@]}")"
  MEDIAN_GFLOPS="$(median_of "${ROUND_GFLOPS[@]}")"
  MEDIAN_TOPS="$(median_of "${ROUND_TOPS[@]}")"
  RUN_LOG=""
  append_csv median median "$MEDIAN_US" "$MEDIAN_GFLOPS" "$MEDIAN_TOPS"
  printf 'Median for %sx%sx%s: %.3f TOPS\n' "$M" "$K" "$N" "$MEDIAN_TOPS"

  if ! awk -v measured="$MEDIAN_TOPS" -v required="$MIN_TOPS" \
      'BEGIN { exit !(measured + 0 >= required + 0) }'; then
    echo "Performance gate failed: $MEDIAN_TOPS < $MIN_TOPS TOPS" >&2
    echo "The NPU run completed; this is not a driver-wedge indication." >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run complete: ${#ARTIFACTS[@]} command(s); the NPU was not touched."
else
  echo
  echo "All ${#ARTIFACTS[@]} generated kernel pair(s) passed."
  echo "CSV results: $OUTPUT"
  echo "Logs: $LOG_DIR"
fi
