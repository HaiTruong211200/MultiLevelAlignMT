import csv
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
import numpy as np
import copy
import qwen.process_data.collator as collator
import qwen.utils.utils as utils
from qwen.process_data.process_data import (
    load_mmt_dataset, 
    process_mmt_data_for_seq2seq_ver2,
)
import re
from peft import LoraConfig, TaskType, get_peft_model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from qwen.training.trainer import CustomSeq2SeqTrainer

import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    NllbTokenizer
)
from transformers.trainer_utils import get_last_checkpoint

# from qwen.models.enc_dec import QwenCrossAttentionEncDec, QwenCrossAttentionEncDecNLLB, print_train_module
from qwen.models.llm_seq2seq import QwenForSeq2SeqConfig, QwenModelForSeq2Seq, print_train_module
from qwen.config.args import DataTrainingArguments, ModelArguments

from qwen.utils.check_weight import check_weight
from qwen.utils.initialize_model_weight import manual_fix_connector_weights

logger = logging.getLogger(__name__)

def main():
    # 1. PARSE ARGUMENTS
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if getattr(model_args, "use_auth_token", None) is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if getattr(model_args, "token", None) is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only `token`.")
        model_args.token = model_args.use_auth_token

    # 2. SETUP LOGGING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters: {training_args}")

    # 3. DETECT CHECKPOINT
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected. Resuming training at {last_checkpoint}. "
                "To train from scratch, change `--output_dir` or add `--overwrite_output_dir`."
            )

    # 4. INITIALIZATION & DATA ARGS
    set_seed(training_args.seed)
    
    pairs = set(data_args.language_pairs.split(",")) if data_args.language_pairs else set()
    languages = set(data_args.languages.split(",")) if data_args.languages else set()
    trans_task = data_args.trans_task.split(",") if data_args.trans_task else []
    
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training language pairs: {pairs} | Translation task: {trans_task} | Languages: {languages}")
    logger.info(f"OT loss lambda: {model_args.ot_lambda}, contrastive loss lambda: {model_args.contrastive_lambda}")

    # 5. CONFIGURATION & TOKENIZERS
    base_config = Qwen2Config.from_pretrained(model_args.llm_path, trust_remote_code=model_args.trust_remote_code)
    print("Model contrastive_lambda:", model_args.contrastive_lambda)
    print("Model contrastive_temperature:", model_args.contrastive_temperature)
    print("Model ot_lambda:", model_args.ot_lambda)
    
    config = QwenForSeq2SeqConfig(
        **base_config.to_dict(),
        mt_model_path=model_args.mt_model_path,
        llm_path=model_args.llm_path,
        num_connector_layers=model_args.num_connector_layers,
        connector_hidden_size=model_args.connector_hidden_size,
        connector_intermediate_size=model_args.connector_intermediate_size,
        connector_num_attention_heads=model_args.connector_num_attention_heads,
        connector_num_key_value_heads=model_args.connector_num_key_value_heads,
        connector_model_method=model_args.connector_model_method,
        fuse_model_group_size=model_args.fuse_model_group_size,
        contrastive_lambda=model_args.contrastive_lambda,
        contrastive_temperature=model_args.contrastive_temperature,
        ot_lambda=model_args.ot_lambda,
        ot_reg=model_args.ot_reg,
        ot_num_iters=model_args.ot_num_iters,
        ot_eps=model_args.ot_eps,
    )

    llm_tokenizer = utils.load_tokenizer(data_args, model_args, training_args, logger, add_eos_token=True)
    seq2seq_tokenizer = NllbTokenizer.from_pretrained(model_args.mt_model_path, trust_remote_code=model_args.trust_remote_code)

    # 6. MODEL SETUP
    if model_args.model_method != "SailorED":
        raise NotImplementedError(f"Model method '{model_args.model_method}' is not implemented yet!")

    is_init = (model_args.run_mode == "init")

    # Xử lý Wandb an toàn (report_to có thể là string hoặc list)
    if training_args.report_to == "wandb" or (isinstance(training_args.report_to, list) and "wandb" in training_args.report_to):
        wandb.init(
            project="Low-Resource-Machine-Translation",
            name="SailorED-pretrain" if is_init else "SailorED-sft"
        )

    if is_init:
        logger.info("Initializing model with pre-trained weights...")
        model = QwenModelForSeq2Seq(config=config, is_init=is_init)
    else:
        logger.info("Loading model from checkpoint...")
        model = QwenModelForSeq2Seq.from_pretrained(model_args.model_name_or_path)

    model.freeze_model(
        freeze_llm=model_args.freeze_llm,
        freeze_decoder=model_args.freeze_decoder,
        freeze_decoder_cross_attn=model_args.freeze_decoder_cross_attn,
        freeze_mt_lm_head=model_args.freeze_mt_lm_head
    )

    # 7. LORA CONFIGURATION
    target_modules = []
    if model_args.train_lora_llm:
        target_modules.append(r"llm\.model\.layers\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    if model_args.train_lora_mt:
        # target_modules.append(r"mt_model\.model\.decoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$")
        target_modules.extend([
            r"mt_model\.model\.decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)$",
            r"mt_model\.model\.decoder\.layers\.\d+\.(fc1|fc2)$"
        ])

    if target_modules:
        modules_to_save = ["connector", "fuse_model"]
        # if not model_args.freeze_decoder:
        #     modules_to_save.append("mt_model.model.decoder")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules="|".join(target_modules),
            modules_to_save=modules_to_save,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        # if not model_args.freeze_decoder:
        #     for name, param in model.mt_model.get_decoder().named_parameters():
        #         param.requires_grad = True

    # 8. FINALIZE
    print_train_module(model)
    model.llm = utils.set_model_special_tokens(model.llm, model_args.model_name_or_path)
    # print(model.generation_config)
    print("Model contractive_lambda:", model.contrastive_lambda)
    print("Model contrastive_temperature:", model.contrastive_temperature)
    print("Model ot_lambda:", model.ot_lambda)
    print("Model ot_reg:", model.ot_reg)
    print("Model ot_num_iters:", model.ot_num_iters)
    print("Model ot_eps:", model.ot_eps)


    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    # Preprocessing data
    # Tokenize dataset
    # ==========================================
    # 1. PREPROCESSING DATA
    # ==========================================
    train_datasets, eval_datasets, test_datasets = None, None, None

    if data_args.mmt_data_path is not None:
        train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(
            pairs, trans_task, data_args, model_args, training_args, logger
        )
        train_datasets, eval_datasets, test_datasets = process_mmt_data_for_seq2seq_ver2(
            train_raw_data, valid_raw_data, test_raw_data, pairs, llm_tokenizer, seq2seq_tokenizer, data_args, training_args
        )
    else:
        logger.warning("`mmt_data_path` is None. Datasets are not loaded.")

    # ==========================================
    # 2. DATA COLLATOR
    # ==========================================
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else llm_tokenizer.pad_token_id
    data_collator = collator.DataCollatorForQwenNLLB(
        llm_tokenizer,
        seq2seq_tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # ==========================================
    # 3. TRAINER INITIALIZATION
    # ==========================================
    check_weight(model)

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=llm_tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.patience)],
    )

    logger.info(model)
    if training_args.do_train:
        utils.print_trainable_parameters(model)

    # ==========================================
    # 4. TRAINING
    # ==========================================
    if training_args.do_train:
        # Xác định checkpoint chính xác
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=None)

        model = model.to(torch.bfloat16)
        trainer.save_model()  

        metrics = train_result.metrics
        
        num_train_samples = len(train_datasets) if train_datasets else 0
        max_train_samples = data_args.max_train_samples or num_train_samples
        metrics["train_samples"] = min(max_train_samples, num_train_samples)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ==========================================
    # 5. CLEANUP
    # ==========================================
    if training_args.report_to == "wandb" or (isinstance(training_args.report_to, list) and "wandb" in training_args.report_to):
        wandb.finish()

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    # model.to("cuda:0")

    # Generate predictions for test set
    predict_tasks = data_args.predict_task.split(",")
    if training_args.do_predict:
        lg_pairs = sorted(test_datasets.keys())
        for lg_pair in lg_pairs:
            cur_test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            for task in cur_test_dataset.keys():
                print(f"\n\n{'='*30}\nPredicting for language pair: {lg_pair}, task: {task}\n{'='*30}")
                if task not in predict_tasks:
                    logger.info(f"skip predict {lg_pair}.{task}")
                    continue
                task_test_dataset = cur_test_dataset[task]
                start = time.time()
                logger.info(f"*** Prediction for {lg_pair}.{task} ***")

                batch_size = training_args.per_device_eval_batch_size
                all_inputs = []
                all_predictions = []
                dataloader = DataLoader(
                    task_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=data_collator   
                )

                for batch in tqdm(dataloader, desc="Generating"):
                    inputs = {
                        k: v.to(model.device)
                        for k, v in batch.items()
                        if k != "labels"
                    }


                    with torch.no_grad():
                        generated_ids = model(**inputs)
                        input_ids, decoder_generate_ids_list = generated_ids

                        # ✅ giữ tensor, không phá structure
                        all_inputs.extend(input_ids.detach().cpu())
                        all_predictions.extend(decoder_generate_ids_list.detach().cpu())


                end = time.time()
                logger.info(f"Prediction completed in {end - start:.2f} seconds.")

                # all_inputs = np.where(all_inputs != -100,all_inputs,llm_tokenizer.pad_token_id)
                # all_predictions = np.where(all_predictions != -100,all_predictions,seq2seq_tokenizer.pad_token_id)
                all_inputs = [x.masked_fill(x == -100, llm_tokenizer.pad_token_id) for x in all_inputs]

                decoded_inputs = [
                    llm_tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "")
                    for x in all_inputs
                ]

                decoded_predictions = [
                    seq2seq_tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "")
                    for x in all_predictions
                ]
                decode_dir = os.path.join(training_args.output_dir, "decode_result")
                os.makedirs(decode_dir, exist_ok=True)
                predic_file_name = f"test.{src_lang}-{tgt_lang}"
                # if task == "general_trans":
                #     predic_file_name += f"-{data_args.test_dataname}"
                output_prediction_file = os.path.join(decode_dir, predic_file_name + ".csv")
                with open(output_prediction_file, "w", encoding="utf-8", newline="") as writer:
                    csv_writer = csv.writer(writer)
                    csv_writer.writerow(["input", "prediction"])
                    for src, pred in zip(decoded_inputs, decoded_predictions):
                        csv_writer.writerow([src, pred])

    

if __name__ == "__main__":
    main()