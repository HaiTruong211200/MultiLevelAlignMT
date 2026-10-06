
#! /bin/bash
set -eux

ROOT_DIR=$(dirname $(dirname `readlink -f $0`))


# ====== ADD DÒNG NÀY ======
export PYTHONPATH=$ROOT_DIR/src

# =========================
export PYTHONUNBUFFERED=1
export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"

config_file=$ROOT_DIR/configs/config.yaml

## model
model_dir="Qwen/Qwen2.5-0.5B-Instruct"
llm_path="Qwen/Qwen2.5-0.5B-Instruct"
mt_model_path="facebook/nllb-200-distilled-600M"
run_mode="init"

freeze_llm=True
freeze_decoder=True
freeze_decoder_cross_attn=True
freeze_mt_lm_head=False
train_lora_llm=False
train_lora_mt=False

model_method="SailorED"
num_connector_layers=4
connector_hidden_size=1024
connector_intermediate_size=4096
connector_num_attention_heads=8
connector_num_key_value_heads=8
connector_model_method="stack"
fuse_model_group_size=4
tag=QwenNLLB_s1

contrastive_lambda=0
ot_lambda=0

## data
language_pairs=vi-en
languages=vi,lo,km
mmt_data_path=$ROOT_DIR/data
trans_task="general_trans"
epoch=2
batch_size=2
gradient_accumulation=2

## save
output_dir=$ROOT_DIR/exps/$tag
mkdir -p $output_dir
cp $0 $output_dir

# ====== SỬA PATH SCRIPT ======
accelerate launch --config_file $config_file \
  $ROOT_DIR/src/qwen/training/train_llm_seq2seq.py \
    --model_name_or_path $model_dir \
    --mt_model_path $mt_model_path \
    --resume_from_checkpoint None \
    --num_connector_layers $num_connector_layers \
    --connector_hidden_size $connector_hidden_size \
    --connector_intermediate_size $connector_intermediate_size \
    --connector_num_attention_heads $connector_num_attention_heads \
    --connector_num_key_value_heads $connector_num_key_value_heads \
    --connector_model_method $connector_model_method \
    --fuse_model_group_size $fuse_model_group_size \
    --model_method ${model_method:-"norm"} \
    --run_mode ${run_mode:-""} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --languages $languages \
    --use_fast_tokenizer \
    --do_eval \
    --do_train \
    --do_predict \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end False \
    --cache_dir ./cache \
    --dataloader_num_workers 24 \
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
    --max_new_tokens 256 \
    --eval_strategy steps \
    --save_strategy no \
    --logging_strategy steps \
    --eval_steps 2000 \
    --logging_steps 10 \
    --save_total_limit 1 \
    --bf16 True \
    --fp16 False \
    --seed 42 \
    --report_to "none" \
    --overwrite_output_dir True \
    --contrastive_lambda $contrastive_lambda \
    --ot_lambda $ot_lambda \
    --llm_path $llm_path \
    --freeze_llm $freeze_llm \
    --freeze_decoder $freeze_decoder \
    --freeze_decoder_cross_attn $freeze_decoder_cross_attn \
    --freeze_mt_lm_head $freeze_mt_lm_head \
    --train_lora_llm $train_lora_llm \
    --train_lora_mt $train_lora_mt \
  | tee $output_dir/train.log
