
#!/usr/bin/env bash
ps -o pid,pgid,comm,args -u "$USER" | egrep 'results_assess_[0-9].sh'
chmod +x c_assess_inf/hpc/results_assess_1.sh

done=$(grep -c '\[done\] .* fully processed' logs/boundary_results_gemini_2_5_flash_preview_05_20.logs)
echo "$done / 500  ($(printf '%0.1f' "$(bc -l <<< "$done*100/500")") %)"

done=$(grep -c '\[done\] .* fully processed' logs/length_results_gemini_2_5_flash_preview_05_20.logs)
echo "$done / 500  ($(printf '%0.1f' "$(bc -l <<< "$done*100/500")") %)"

# run all 11 types
  caffeinate -dimsu ./c_assess_inf/hpc/results_assess_1.sh

# run only specific types
  caffeinate -dimsu ./c_assess_inf/hpc/results_assess_1.sh boundary language style tone


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

