import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
import time

import datasets
import wandb
import numpy as np
import copy
import qwen.process_data.collator as collator
import qwen.utils.utils as utils
from qwen.process_data.process_data import load_mmt_dataset, process_mmt_data_for_seq2seq, load_data_pretrain, process_pretrain_data_for_seq2seq
import re
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

import transformers
from transformers import (
    AutoConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from qwen.models.enc_dec import QwenCrossAttentionEncDec
from qwen.config.args import DataTrainingArguments, ModelArguments

from qwen.utils.check_weight import check_weight
from qwen.utils.initialize_model_weight import manual_fix_connector_weights

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    logger.info(f"Data arguments {data_args}")

    pairs = set(data_args.language_pairs.split(","))
    languages = set(data_args.languages.split(","))
    trans_task = data_args.trans_task.split(",")
    logger.info(f"Training lanauage pairs: {pairs}\nTraining translation task: {trans_task} Training languages: {languages}")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code
    )

    add_eos_token = True if model_args.model_method == "default" else False
    tokenizer = utils.load_tokenizer(data_args, model_args, training_args, logger, add_eos_token=add_eos_token)


    model = QwenCrossAttentionEncDec.from_pretrained(model_args.model_name_or_path, config=config)

    
    # model.generation_config = GenerationConfig.from_pretrained(
    #     model_args.model_name_or_path,
    # )

    model = utils.set_model_special_tokens(model, model_args.model_name_or_path)
    print(model.generation_config)

    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    ## Preprocessing data
    ## Tokenize dataset
    if data_args.mmt_data_path is not None:
        if model_args.run_mode == "init":
            train_raw_data, valid_raw_data, test_raw_data = load_data_pretrain(languages, data_args, model_args, training_args,logger)
            train_datasets, eval_datasets, test_datasets = process_pretrain_data_for_seq2seq(train_raw_data, valid_raw_data, test_raw_data, languages, tokenizer, data_args, training_args)
        elif model_args.run_mode == "sft":
            train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, trans_task, data_args, model_args, training_args, logger)
            train_datasets, eval_datasets, test_datasets = process_mmt_data_for_seq2seq(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, data_args, training_args)
    ## Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        if model_args.model_method == "default":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None,
            )
        elif model_args.model_method in ["SailorED"]:
            data_collator = collator.DataCollatorForLamate(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if training_args.fp16 else None
            )
        else:
            print("Not implement this model yet!")
            exit()

    optimizer = None


    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    predict_tasks = data_args.predict_task.split(",")
    if training_args.do_predict:
        lg_pairs = sorted(test_datasets.keys())
        for lg_pair in lg_pairs:
            cur_test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            for task in cur_test_dataset.keys():
                if task not in predict_tasks:
                    logger.info(f"skip predict {lg_pair}.{task}")
                    continue
                task_test_dataset = cur_test_dataset[task]
                start = time.time()
                logger.info(f"*** Prediction for {lg_pair}.{task} ***")

                predict_results = trainer.predict(
                    task_test_dataset, 
                    metric_key_prefix="test", 
                    num_beams=num_beams, 
                    max_new_tokens=data_args.max_new_tokens,
                    do_sample=model_args.do_sample
                )
                metrics = predict_results.metrics

                if int(torch.cuda.current_device()) == 0:
                    predictions = predict_results.predictions
                    if len(predictions) != len(task_test_dataset):
                        predictions = predictions[:len(task_test_dataset)]
                    num_tokens = sum([ len(t) for t in predictions ])
                    timediff = time.time() - start
                    logger.info(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
                    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.replace("\n", "") for pred in predictions]
                    
                    decode_dir = os.path.join(training_args.output_dir, "decode_result")
                    os.makedirs(decode_dir, exist_ok=True)
                    predic_file_name = f"test-{src_lang}-{tgt_lang}-{task}"
                    if task == "general_trans":
                        predic_file_name += f"-{data_args.test_dataname}"
                    output_prediction_file = os.path.join(decode_dir, predic_file_name)
                    with open(output_prediction_file, "w", encoding="utf-8") as writer:
                        writer.write("\n".join(predictions))

if __name__ == "__main__":
    main()