#!/usr/bin/env bash
# Lint, types and tests, in one command. Run before committing, and while working
# through a refactor -- the three together are the gate, since none catches another's
# class of defect: ruff sees a name that is never read, mypy sees a field that cannot
# exist, pytest sees a value that is wrong.
#
#   scripts/check.sh              all three
#   scripts/check.sh lint         ruff only  (instant)
#   scripts/check.sh types        mypy only  (fast, run this while editing)
#   scripts/check.sh tests        pytest only
#
# Extra arguments after the target are passed through, so a focused loop works:
#   scripts/check.sh types src/stemify/training
#   scripts/check.sh tests tests/training/test_checkpoint.py -x
#
# Deliberately *not* fail-fast when running both: a red type check should not hide a
# red test suite. One run, the whole picture.
set -uo pipefail
cd "$(dirname "$0")/.."

target="${1:-all}"
[ $# -gt 0 ] && shift

lint=0
types=0
tests=0

run_lint() {
  echo "── ruff ─────────────────────────────────────────"
  # `check`, never `--fix` here. A gate that edits your working tree while reporting on
  # it turns a red run into a diff you did not write, and the fix for an unused import
  # is sometimes to *use* it.
  uv run ruff check "$@" 2>&1
  return "${PIPESTATUS[0]}"
}

run_types() {
  echo "── mypy ─────────────────────────────────────────"
  # No truncation. An earlier version piped through `tail -40`, which silently hid all
  # but the last few errors -- and then a count taken from that output was wrong by 3x
  # while looking perfectly plausible. A backlog you are working through is exactly
  # when you need the whole list; pipe it yourself if you want less.
  uv run mypy "$@" 2>&1
  return "${PIPESTATUS[0]}"
}

run_tests() {
  echo "── pytest ───────────────────────────────────────"
  # No -q here: pyproject's addopts already sets it, and a second one makes -qq, which
  # suppresses the summary line -- so the run ends with no pass/fail count at all and
  # you go hunting for it with --collect-only. -p no:warnings keeps librosa and
  # mir_eval's deprecation noise from burying that line.
  uv run pytest -p no:warnings "$@" 2>&1 | grep -vE "^\s*$" | tail -25
  return "${PIPESTATUS[0]}"
}

case "$target" in
  lint)  run_lint "$@"; lint=$? ;;
  types) run_types "$@"; types=$? ;;
  tests) run_tests "$@"; tests=$? ;;
  all)   run_lint; lint=$?; echo; run_types; types=$?; echo; run_tests; tests=$? ;;
  *)     echo "usage: $0 [all|lint|types|tests] [args...]" >&2; exit 2 ;;
esac

echo
if [ "$lint" -ne 0 ] || [ "$types" -ne 0 ] || [ "$tests" -ne 0 ]; then
  echo "FAIL  (ruff=$lint mypy=$types pytest=$tests)"
  exit 1
fi
echo "PASS"
