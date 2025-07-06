#!/usr/bin/env bash
target=$(readlink -f "$1")          # canonical path
for pid in /proc/[0-9]*; do
  fdpath="$pid/fd"
  [[ -d $fdpath ]] || continue
  for fd in "$fdpath"/*; do
    link=$(readlink -f "$fd" 2>/dev/null) || continue
    if [[ $link == "$target" ]]; then
      ps -o pid,pgid,sid,comm --no-headers "${pid#/proc/}"
      break
    fi
  done
done
