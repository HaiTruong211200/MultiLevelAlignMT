
#! /bin/bash
set -eux

ROOT_DIR=$(dirname $(dirname `readlink -f $0`))


# ====== ADD DÒNG NÀY ======
export PYTHONPATH=$ROOT_DIR/src

# =========================
export PYTHONUNBUFFERED=1
export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"

config_file=$ROOT_DIR/configs/myconfig.yaml

## model
model_dir="/kaggle/input/notebooks/truonghai/train-sailored-stage1/mt/exps/QwenED_s1"
run_mode="sft"

model_method="SailorED"
encoder_method="stack"
encoder_layer_num=8
decoder_layer_num=8
decoder_hidden_size=672
decoder_intermediate_size=2560
decoder_num_attention_heads=8
decoder_num_key_value_heads=8
lora_r=32
lora_alpha=64
# decoder_param_method="freeze"
tag=QwenED_s2
contrastive_lambda=1
contrastive_temperature=0.08
ot_lambda=4
num_layers_align=1

## data
language_pairs=km-vi,lo-vi
mmt_data_path=/kaggle/input/datasets/truonghai/data-stage2
trans_task="general_trans"
epoch=2
batch_size=12
gradient_accumulation=2

## save
output_dir=$ROOT_DIR/exps/$tag
mkdir -p $output_dir
cp $0 $output_dir

# ====== SỬA PATH SCRIPT ======
accelerate launch --config_file $config_file \
  $ROOT_DIR/src/qwen/training/train_sailored.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint None \
    --encoder_layer_num ${encoder_layer_num} \
    --decoder_layer_num $decoder_layer_num \
    --decoder_hidden_size $decoder_hidden_size \
    --decoder_intermediate_size $decoder_intermediate_size \
    --decoder_num_attention_heads $decoder_num_attention_heads \
    --decoder_num_key_value_heads $decoder_num_key_value_heads \
    --encoder_method $encoder_method \
    --model_method ${model_method:-"norm"} \
    --run_mode ${run_mode:-""} \
    --decoder_param_method ${decoder_param_method:-"share"} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_eval \
    --do_train \
    --do_predict \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end False \
    --cache_dir ./cache \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir  $output_dir \
    --num_train_epochs $epoch \
    --patience 3 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 512 \
    --eval_strategy steps \
    --save_strategy no \
    --logging_strategy steps \
    --eval_steps 2000 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --fp16 False \
    --seed 42 \
    --report_to "none" \
    --overwrite_output_dir True \
    --contrastive_lambda $contrastive_lambda \
    --contrastive_temperature $contrastive_temperature \
    --num_layers_align $num_layers_align \
    --ot_lambda $ot_lambda \
  | tee $output_dir/train.log
