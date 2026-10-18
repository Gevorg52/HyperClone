#!/usr/bin/env bash
# WP-F5 sequential chains (two run concurrently on the 96 GB card).
#   bash _infusion/wpf5_chain.sh A   -> laptop bil, pp, ppc, then 360 base, bil, pp, ppc
#   bash _infusion/wpf5_chain.sh B   -> gpu base, bil, pp, ppc
# Laptop `base` is NOT re-run: it is the WP-B2 run `_dataset/Laptop/.../opreg0/b2` (same venv,
# same argv as wpf5_run.sh laptop base).
cd "$(dirname "$0")/.."
case "$1" in
  A) JOBS="laptop:bil laptop:pp laptop:ppc 360:base 360:bil 360:pp 360:ppc" ;;
  B) JOBS="gpu:base gpu:bil gpu:pp gpu:ppc" ;;
esac
for j in $JOBS; do
  echo "[chain $1] start $j $(date)"; bash _infusion/wpf5_run.sh "${j%%:*}" "${j##*:}" > /dev/null 2>&1
  echo "[chain $1] end $j rc=$? $(date)"
done
echo "[chain $1] ALL DONE $(date)"
