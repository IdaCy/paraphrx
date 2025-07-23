# runs track BORG

got before:

all data:
- 1-2 & 1-3, all layers
- attn dropout 0
- bf16

alpaca:
- 1-2 all layers, & 1-3, specific layers
- attn dropout 0
- bf16

alpaca:
- 1-3 specific layers
- lf 1.8 (1.5?) e-4
- GAs 32
- sas 500

## alpaca buckets 1-2

### A-base

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-2_A-base" \
  --run_name A-base_buckets_1-2_all_layers \
  --buckets 1-2 \
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

### A-hiLR

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-2_A-hiLR" \
  --run_name A-hiLR_buckets_1-2_all_layers \
  --buckets 1-2 \
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
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-2_A-long" \
  --run_name A-long_buckets_1-2_all_layers \
  --buckets 1-2 \
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


# A extra
# Alpaca full-parameter fine-tune, buckets 1-3, conservative LR
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_6/alpaca_all_layers/outputs_buckets_1-3_A-base-plus" \
  --run_name A-base-plus_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2.5e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 \
  --save_steps 500 \
  $WANDB_FLAG







## alpaca buckets 1-3

### B-qlora-16

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-3_B-qlora-16" \
  --run_name B-qlora-16_buckets_1-3_specific_layers \
  --buckets 1-3 \
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

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-3_B-qlora-32" \
  --run_name B-qlora-32_buckets_1-3_specific_layers \
  --buckets 1-3 \
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

# B-qlora-lowLR

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-3_B-qlora-lowLR" \
  --run_name B-qlora-lowLR_buckets_1-3_specific_layers \
  --buckets 1-3 \
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



# B extra

# Alpaca QLoRA, buckets 1-3, lower LR & higher WD for stability
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_specific_layers/outputs_buckets_1-3_B-qlora-lowLR-8e-5" \
  --run_name B-qlora-lowLR-8e-5_buckets_1-3_specific_layers \
  --buckets 1-3 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 \
  --save_steps 500 \
  $WANDB_FLAG





## all data buckets 1-2

### C-base

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-2_C-base" \
  --run_name C-base_buckets_1-2_all_layers \
  --buckets 1-2 \
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
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-2_C-lowLR-long" \
  --run_name C-lowLR-long_buckets_1-2_all_layers \
  --buckets 1-2 \
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
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-2_C-hiLR" \
  --run_name C-hiLR_buckets_1-2_all_layers \
  --buckets 1-2 \
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

# C extra

# All-data full-parameter fine-tune baseline, buckets 1-2, very conservative LR, fewer epochs
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_all_layers/outputs_buckets_1-2_C-lowLR-short" \
  --run_name C-lowLR-short_buckets_1-2_all_layers \
  --buckets 1-2 \
  --bf16 \
  --target_modules none \
  --batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 2 \
  --eval_steps 500 \
  --save_steps 500 \
  $WANDB_FLAG







## all data buckets 1-3

### D-qlora-16

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-3_D-qlora-16" \
  --run_name D-qlora-16_buckets_1-3_specific_layers \
  --buckets 1-3 \
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
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-3_D-qlora-32" \
  --run_name D-qlora-32_buckets_1-3_specific_layers \
  --buckets 1-3 \
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

### D-qlora-8-lowLR

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-3_D-qlora-8-lowLR" \
  --run_name D-qlora-8-lowLR_buckets_1-3_specific_layers \
  --buckets 1-3 \
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




### D extra

# All-data (alpaca+gsm8k+mmlu) QLoRA, buckets 1-3, mid learning rate

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-3_D-qlora-midLR-1p2e-4" \
  --run_name D-qlora-midLR-1p2e-4_buckets_1-3_specific_layers \
  --buckets 1-3 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1.2e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 \
  --save_steps 500 \
  $WANDB_FLAG
###### DONE #################



# E extra

# OPTIONAL: test whether adding bucket-4 noisy paraphrases improves robustness
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/all_data_specific_layers/outputs_buckets_1-4_E-qlora-robust-8e-5" \
  --run_name E-qlora-robust-8e-5_buckets_1-4_specific_layers \
  --buckets 1-4 \
  --bf16 \
  --bnb_8bit_optim \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 \
  --save_steps 500 \
  $WANDB_FLAG



# others

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_6/all_data_specific_layers/1-3_midLR_qlora" \
  --run_name 1-3_midLR_qlora_ft \
  --buckets 1-3 \
  --bf16 --bnb_8bit_optim \
  --lora_rank 16 --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 --gradient_accumulation_steps 4 \
  --learning_rate 1.2e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 --save_steps 500 \
  $WANDB_FLAG
###### DONE #################

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_6/alpaca_specific_layers/1-3_lowLR_qlora" \
  --run_name 1-3_lowLR_qlora_ft \
  --buckets 1-3 \
  --bf16 --bnb_8bit_optim \
  --lora_rank 16 --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 --save_steps 500 \
  $WANDB_FLAG
###### DONE #################

###### DONE #################
RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_6/all_data_all_layers/1-2_lowLR_fullFT" \
  --run_name 1-2_lowLR_fullFT_ft \
  --buckets 1-2 \
  --bf16 --target_modules none --bnb_8bit_optim \
  --batch_size 6 --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 2 \
  --eval_steps 500 --save_steps 500 \
  $WANDB_FLAG
###### DONE #################


RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_6/all_data_specific_layers/1-4_lowLR_qlora" \
  --run_name 1-4_lowLR_qlora_ft \
  --buckets 1-4 \
  --bf16 --bnb_8bit_optim \
  --lora_rank 16 --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 2 --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 500 --save_steps 500 \
  $WANDB_FLAG
