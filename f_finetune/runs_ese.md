# runs track ESE HPC

got already:
- all in 1st run

2nd:
- 1-1 & -2, all data, all/specific layers
- 1-1 -5, all data
- 1-1 -5 alpaca

1-3 all data
- lr 1.8e-4
- GA 16
- sas 400

1-3 all data
- lr 1e-4
- GA 16
- sas 300



## alpaca buckets 1-3

### A-base

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-3_A-base" \
  --run_name A-base_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

### A-base-mem

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_6/alpaca_all_layers/outputs_buckets_1-3_A-base-mem" \
  --run_name A-base-mem_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

### A-hiLR

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-3_A-hiLR" \
  --run_name A-hiLR_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

### A-long

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-3_A-long" \
  --run_name A-long_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 4 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

## alpaca buckets 1-2

### B-qlora-16

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-2_B-qlora-16" \
  --run_name B-qlora-16_buckets_1-2_specific_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 1000 \
  --save_steps 1000 \
  $WANDB_FLAG
###### DONE #################

### B-qlora-32

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-2_B-qlora-32" \
  --run_name B-qlora-32_buckets_1-2_specific_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 1000 \
  --save_steps 1000 \
  $WANDB_FLAG
###### DONE #################

# B-qlora-lowLR

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-2_B-qlora-lowLR" \
  --run_name B-qlora-lowLR_buckets_1-2_specific_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 1000 \
  --save_steps 1000 \
  $WANDB_FLAG
###### DONE #################

## all data buckets 1-3

### C-base

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-3_C-base" \
  --run_name C-base_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

### C-lowLR-long

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-3_C-lowLR-long" \
  --run_name C-lowLR-long_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 4 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

### C-hiLR

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-3_C-hiLR" \
  --run_name C-hiLR_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 1 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

## all data buckets 1-2

### D-qlora-16

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-2_D-qlora-16" \
  --run_name D-qlora-16_buckets_1-2_specific_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 1000 \
  --save_steps 1000 \
  $WANDB_FLAG
###### DONE #################

### D-qlora-32

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-2_D-qlora-32" \
  --run_name D-qlora-32_buckets_1-2_specific_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2.5e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 1000 \
  --save_steps 1000 \
  $WANDB_FLAG
###### DONE #################

### D-qlora-8-lowLR

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-2_D-qlora-8-lowLR" \
  --run_name D-qlora-8-lowLR_buckets_1-2_specific_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 1000 \
  --save_steps 1000 \
  $WANDB_FLAG
###### DONE #################



