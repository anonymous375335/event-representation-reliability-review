#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${repo_root}/data/cms_h4l_open_data_manifest.csv"
dest="${repo_root}/data/cms_h4l_2012_reduced_nanoaod/root_files"
kind_filter="${1:-all}"

mkdir -p "${dest}"

if [[ ! -f "${manifest}" ]]; then
  echo "Missing manifest: ${manifest}" >&2
  exit 1
fi

echo "Destination: ${dest}"
echo "Filter: ${kind_filter}"

tail -n +2 "${manifest}" | while IFS=, read -r sample kind record_id record_url filename size_bytes adler32 http_url xrootd_url role; do
  case "${kind_filter}" in
    all) ;;
    mc) [[ "${kind}" == mc_* ]] || continue ;;
    data) [[ "${kind}" == "observed_data" ]] || continue ;;
    *) echo "Unknown filter '${kind_filter}'. Use: all, mc, or data." >&2; exit 1 ;;
  esac

  target="${dest}/${filename}"
  if [[ -f "${target}" ]]; then
    actual_bytes="$(wc -c < "${target}" | tr -d ' ')"
    if [[ "${actual_bytes}" == "${size_bytes}" ]]; then
      echo "[ok] ${filename} (${actual_bytes} bytes)"
      continue
    fi
    echo "[size-mismatch] ${filename}: have ${actual_bytes}, expected ${size_bytes}; resuming"
  fi

  echo "[download] ${sample} ${record_url}"
  curl -L --fail --show-error --continue-at - --output "${target}" "${http_url}"
  actual_bytes="$(wc -c < "${target}" | tr -d ' ')"
  if [[ "${actual_bytes}" != "${size_bytes}" ]]; then
    echo "[failed-size] ${filename}: have ${actual_bytes}, expected ${size_bytes}" >&2
    exit 1
  fi
  echo "[done] ${filename} (${actual_bytes} bytes; expected adler32 ${adler32})"
done

echo "Download pass finished."
