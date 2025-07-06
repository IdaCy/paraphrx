# Progress Checks & Stops

ps -f -ww -u "$USER" | grep -Ei 'local_modelchoice' | grep -v grep


for p in 14616 18654 76924 95357 16947 68642 69031; do
  echo "=== wrapper $p ==="
  kids=$(pgrep -P "$p")
  if [ -z "$kids" ]; then
    echo "  (no active worker — wrapper is sleeping or finished)"
  else
    ps -o pid,ppid,etime,args -p $(echo "$kids" | tr '\n' ',' | sed 's/,$//')
  fi
  echo
done


for p in 14616 18654 76924 95357 16947 68642 69031; do
  echo "=== wrapper $p ==="
  kids=$(pgrep -P "$p")
  if [ -z "$kids" ]; then
    echo "  (no active worker — wrapper is sleeping or finished)"
  else
    # -o …command (or args) plus -ww to remove width limit
    ps -o pid,ppid,etime,command -ww -p $(echo "$kids" | paste -sd, -)
  fi
  echo
done



## Any in gerneal:
### wrapper = any *.sh still running

for p in $(pgrep -f '.*\.sh'); do
  echo "=== wrapper $p ==="
  kids=$(pgrep -P "$p")
  if [ -z "$kids" ]; then
    echo "  (sleeping or finished)"
  else
    ps -o pid,etime,command -ww -p $(echo "$kids" | paste -sd, -)
  fi
done


pmset -g assertions | grep -A1 PreventUserIdleSystemSleep


ps -o pid,ppid,etime,command -ww -u "$USER" \
  | grep 500_prxed | grep -v grep

  find a_data -name '*_prxed_*.json' -mmin -1 -print | tail -1



ps -o pid,ppid,etime,command -ww -u "$USER" \
  | grep main_500_prxed_diff82_obstruction | grep -v grep

  find a_data -name '*_prxed_*.json' -mmin -1 -print | tail -1




ps -o pid,ppid,etime,command -ww -u "$USER" \
  | grep main_500_prxed_context | grep -v grep

  find a_data -name '*_prxed_*.json' -mmin -1 -print | tail -1



## KILL THE JOBS
for cpid in $(pgrep -f 'caffeinate -dims nohup .*_prx.sh'); do
  pkill -TERM -P "$cpid"; kill -TERM "$cpid"
done
## WILL KILL THEM
