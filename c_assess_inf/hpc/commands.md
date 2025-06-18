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

unpaid 1:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY 


test 2 unpaid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME 


test 3 unpaid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68 


test 4 unpaid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI 


test 5 unpaid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM tone 1 2 3 4 5



paid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc voice 1 2 3 4 5


2nd paid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q


3rd paid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI


4th paid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY


5th paid:
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo




## new plan

unpaid 1:
AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY boundary 1 2 3 4 5

test 2 unpaid:
AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME length 1 2 3 4 5

test 3 unpaid:
AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68 style 1 2 3 4 5

test 4 unpaid:
AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI syntax 1 2 3 4 5

test 5 unpaid:
AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM length 1 2 3 4 5


paid:
AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc style 1 2 3 4 5

2nd paid:
AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q syntax 1 2 3 4 5

3rd paid:
AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI tone 1 2 3 4 5

4th paid:
AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY voice 1 2 3 4 5

5th paid:
AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh -k AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo boundary 1 2 3 4 5


## new plan
need:
- language 5
- special_char 3 4 5
- style 4 5
- tone 5

unpaid 1:         language 5
AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY language 5

test 2 unpaid:
AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME

test 3 unpaid:    style 4
AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68 style 4

test 4 unpaid:    tone 5
AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI tone 5

test 5 unpaid:
AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM


paid:             style 5
AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc style 5

2nd paid:         special_char 3 4 5
AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q special_char 3 4 5

!!!! GSM8K TEST
3rd paid:
AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI boundary

!!!! GSM8K TEST
4th paid:
AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY context

!!!! GSM8K TEST
5th paid:
AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k.sh -k AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo extra_a

## new plan
unpaid 1:         context 3 4 5
AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY context 3 4 5

test 2 unpaid:    extra_b 2 3 4 5
AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME extra_b 2 3 4 5

test 3 unpaid:    language 2 3 5
AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68 language 2 3 5

test 4 unpaid:    length 3 4 5
AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI length 3 4 5

test 5 unpaid:    obstruction 2 3 4 5
AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM obstruction 2 3 4 5


paid:             special_char 1 2 3 4 5
AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc



2nd paid:         style 2 3 4 5
AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q style 2 3 4 5

3rd paid:         syntax 3 4 5
AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI syntax 3 4 5

4th paid:         tone 2 3 4 5
AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY tone 2 3 4 5

5th paid:         voice 3 4 5
AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo voice 3 4 5



## before


### unpaid
context slice 1       UNPAID 1
AIzaSyCQkDVEvgQofmpkmVAbbPwOR2m6SX6wYsY

context slices 2-5    UNPAID 2
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyDw-mbGFmFlYGqB_i2RYFL-KJ3wN10McJM context 2 3 4 5

extra_a slices 1-2    UNPAID 3
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyBXXuGjVs-oCv4HQ7eVDKMnUYC-j1Fk8ME extra_a 1 2

extra_a slices 3-4    UNPAID 4
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68 extra_a 3 4

extra_a slices 5
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyB06EIgeIZ-W5gnKvVXOGKU7CUAjyxf5bI extra_a 5

extra_b
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc extra_b 1 2 3 4 5

language
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q language 1 2 3 4 5

length
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI length 1 2 3 4 5

obstruction
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY obstruction 1 2 3 4 5

style
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo style 1 2 3 4 5

syntax
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyC67Z6W3XEf9wDV0qiVUjxpW4juDjNT3Xo -m "gemini-2.5-pro-preview-06-05" syntax 1 2 3 4 5

special_char
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY -m "gemini-2.5-pro-preview-06-05" special_char 1 2 3 4 5

tone
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI -m "gemini-2.5-pro-preview-06-05" tone 1 2 3 4 5

voice
caffeinate -dimsu ./c_assess_inf/hpc/assess_loop.sh -k AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q -m "gemini-2.5-pro-preview-06-05" voice 1 2 3 4 5

