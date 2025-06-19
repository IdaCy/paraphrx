# commands shortcut for assess results script

## make runnable
chmod +x c_assess_inf/hpc/results_assess_1.sh


## general start
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k KEY_HERE context 2 3 4 5


## check what runs & make runnable
ps -o pid,pgid,comm,args -u "$USER" | egrep 'assess_loop_gsm8k.sh'

ps -o pid,pgid,comm,args -u "$USER" | egrep 'voice_slice1_results_gemini_2_5_flash_preview_05_20'

done=$(grep -c '\[done\] .* fully processed' logs/context_slice1_results_gemini_2_5_flash_preview_05_20.logs)
echo "$done / 100  ($(printf '%0.1f' "$(bc -l <<< "$done*100/100")") %)"

done=$(grep -c '\[done\] .* fully processed' logs/obstruction_results_gemini_2_5_flash_preview_05_20.logs)
echo "$done / 500  ($(printf '%0.1f' "$(bc -l <<< "$done*100/500")") %)"

### check all of them
ps -o pid,pgid,etime,args -u "$USER" | grep '[c]affeinate'  


## check moge & status

for f in logs/*_results_gemini_2_5_flash_preview_05_20.logs(N); do
  slice=${${f:t}%_results*}
  done=$(grep -c '\[done\]' "$f")
  start=$(grep -c '▶ id'   "$f")
  printf "%-12s %4d / %d\n" "$slice" "$done" "$start"
done | sort

for f in logs/*_results_gemini_2_5_flash_preview_05_20.logs(N); do
  done=$(grep -c '\[done\]' "$f")
  percent=$(( done * 100 / 500 ))
  printf "%-12s %3d%%\n" "${${f:t}%_results*}" "$percent"
done


## KILL specific one

pid=$(lsof +wt -- logs/extra_slice1_results_gemini_2_5_flash_preview_05_20.logs)

- First, terminate the whole process-group so the child bash script dies too -(leading minus → “whole group”)
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
kill -TERM -$pgid

- Wait a beat, then verify
sleep 1
lsof +wt -- logs/extra_slice1_results_gemini_2_5_flash_preview_05_20.logs \
  || echo "file is no longer open"

## kill all:
pgrep -fl caffeinate

- 2) Politely ask them all to exit:
killall caffeinate
#-> sends SIGTERM to every process named "caffeinate"

- 3) Give macOS a moment! to clear power-assertion! then confirm:
pgrep -fl caffeinate
#-> should print nothing

- 4) If any remain, force-kill:
killall -9 caffeinate  


# check sleep events
pmset -g log | grep -e "Sleep" -e "Wake"



# planning

## running:

## new plan

done gsm8k q: boundary length style syntax
