#!/usr/bin/env bash
set -euo pipefail

### CONFIGURATION
# Directory to watch:
DIR="c_assess_inf/output50k/activations"

# Where to log everything:
LOGFILE="logs/monitor.log"

# Threshold in bytes (200 GB):
THRESHOLD=$((200 * 1024 * 1024 * 1024))

# How often to check (in seconds):
CHECK_INTERVAL=1200   # 20 minutes

# Total runtime (in seconds):
MAX_RUNTIME=$((10 * 3600))   # 10 hours

# How many checks before a 2 hr summary? (2 hr = 120 min = 6 × 20 min)
SUMMARY_EVERY=6

### INITIALIZATION
start_time=$(date +%s)
iteration=0

# Counters for each 2-hr window:
window_count=0
window_bytes=0

# Totals over entire run:
total_count=0
total_bytes=0

# Flag any unexpected error
unexpected=false

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== STARTING monitor: DIR=$DIR  THRESHOLD=$(numfmt --to=iec $THRESHOLD)  MAX_RUNTIME=10h ==="

### MAIN LOOP
while (( $(date +%s) - start_time < MAX_RUNTIME )); do
  iteration=$((iteration + 1))

  # Measure current directory size
  current_size=$(du -sb "$DIR" 2>/dev/null | cut -f1 || echo 0)
  log "Check #$iteration: directory size is $(numfmt --to=iec $current_size)"

  if (( current_size > THRESHOLD )); then
    log "  !!  Exceeds threshold ($(numfmt --to=iec $THRESHOLD)), deleting oldest files…"
    # Work down from oldest file until under threshold
    # We list files sorted by mtime ascending (oldest first)
    while (( current_size > THRESHOLD )); do
      # find the single oldest file
      oldest=$(find "$DIR" -type f -printf '%T@ %p\n' 2>/dev/null \
               | sort -n | head -n1 | cut -d' ' -f2-)
      [[ -z "$oldest" ]] && {
        log "  !!!  No more files found but still over threshold!"
        unexpected=true
        break
      }
      # get its size
      fsize=$(stat -c%s "$oldest" 2>/dev/null || echo 0)

      # attempt deletion
      if rm -f "$oldest"; then
        log "    ✔ Deleted: $oldest ($(numfmt --to=iec $fsize))"
        (( window_count++, total_count++ ))
        (( window_bytes+=fsize, total_bytes+=fsize ))
        (( current_size -= fsize ))
      else
        log "    XXX FAILED to delete $oldest"
        unexpected=true
        # remove from consideration to avoid infinite loop
        # by renaming it out of the way
        mv "$oldest" "${oldest}.delete-failed" && \
          log "    ! ! !  Renamed to avoid future retry: ${oldest}.delete-failed"
      fi
    done
  else
    log " ✓✔ Under threshold, no deletions needed."
  fi

  # Every SUMMARY_EVERY checks (i.e. ~2 hrs), output a window summary
  if (( iteration % SUMMARY_EVERY == 0 )); then
    if (( window_count > 0 )); then
      log "――  2 hr SUMMARY: deleted $window_count files totaling $(numfmt --to=iec $window_bytes)"
    else
      log "――  2 hr SUMMARY: no deletions were needed in this period"
    fi
    # reset window counters
    window_count=0
    window_bytes=0
  fi

  # Sleep until next check (unless about to finish)
  now=$(date +%s)
  elapsed=$(( now - start_time ))
  time_left=$(( MAX_RUNTIME - elapsed ))
  (( time_left <= CHECK_INTERVAL )) && break
  sleep $CHECK_INTERVAL
done

### FINAL SUMMARY
log "=== FINAL SUMMARY (10 hr run complete) ==="
log "Total files deleted: $total_count"
log "Total bytes freed:    $(numfmt --to=iec $total_bytes)"
if $unexpected; then
  log " ! One or more unexpected errors occurred; see above logs."
else
  log "✓✔ No unexpected errors encountered."
fi

log "=== MONITORING ENDED ==="
