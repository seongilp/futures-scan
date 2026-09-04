#!/usr/bin/env bash
# ebs 크론용: 스캔→차트→백테스트→리플레이 후 deploy 브랜치 publish. 조용히 스킵하지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/scan-$(date +%Y%m%d).log"
export PATH="$HOME/.local/bin:$PATH"
exec 9>"/tmp/futures-scan.lock"
if ! flock -n 9; then echo "[$(date '+%F %T')] [SKIP] 이전 실행이 아직 락 보유" >> "$LOG"; exit 0; fi
cd "$ROOT"
{
  echo "[$(date '+%F %T')] start"
  git fetch -q origin main && git reset -q --hard origin/main
  uv sync -q
  timeout 900 uv run futures-scan all --tf "${SCAN_TF:-1h}" || echo "[ERROR] futures-scan all exit $?"
  bash scripts/publish-out.sh || echo "[ERROR] publish exit $?"
  echo "[$(date '+%F %T')] done"
} >> "$LOG" 2>&1
