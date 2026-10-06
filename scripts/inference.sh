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
model_dir="/kaggle/input/notebooks/truonghai/train-sailored-stage2/mt/exps/QwenED_s2"

model_method="SailorED"

## data
language_pairs=km-vi,lo-vi
mmt_data_path=/kaggle/input/datasets/truonghai/data-stage2
trans_task="general_trans"
batch_size=16
gradient_accumulation=2

## save
output_dir=$ROOT_DIR/exps/$tag
mkdir -p $output_dir
cp $0 $output_dir

# ====== SỬA PATH SCRIPT ======
accelerate launch --config_file $config_file \
  $ROOT_DIR/src/qwen/training/train_sailored.py \
    --model_name_or_path $model_dir \
    --model_method ${model_method:-"norm"} \
    --run_mode ${run_mode:-""} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_predict \
    --cache_dir ./cache \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --output_dir  $output_dir \
    --num_train_epochs $epoch \
    --patience 3 \
    --per_device_eval_batch_size $batch_size \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 512 \
    --bf16 True \
    --fp16 False \
    --seed 42 \
  | tee $output_dir/train.log
