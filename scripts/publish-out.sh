#!/usr/bin/env bash
# out/ 를 orphan 커밋 하나로 `deploy` 브랜치에 force-push 한다.
# Vercel 프로덕션 브랜치가 deploy 라 푸시 즉시 정적 배포된다. 히스토리를 쌓지 않아
# 매시간 돌려도 리포가 비대해지지 않는다(이전 커밋은 unreachable → GC).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/out"
REMOTE="${PUBLISH_REMOTE:-origin}"
[ -f "$OUT/index.html" ] || { echo "out/index.html 없음 — 먼저 scan/chart 실행" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$ROOT" worktree add --detach "$WORK" HEAD >/dev/null 2>&1 || true
rm -rf "$WORK"/* "$WORK"/.gitignore 2>/dev/null || true
cp -R "$OUT"/. "$WORK"/
rm -rf "$WORK/.vercel"
git -C "$WORK" checkout -q --orphan deploy
git -C "$WORK" add -A
git -C "$WORK" -c user.name=futures-scan-bot -c user.email=bot@futures-scan commit -q -m "publish $(date -u +%FT%TZ)"
git -C "$WORK" push -q --force "$REMOTE" deploy:deploy
git -C "$ROOT" worktree remove --force "$WORK" 2>/dev/null || true
echo "published deploy @ $(date -u +%FT%TZ)"
