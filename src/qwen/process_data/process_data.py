#!/usr/bin/env python
# coding=utf-8

import copy
import os
from typing import Optional
import torch
from datasets import load_dataset, concatenate_datasets
import glob
import regex
import random
random.seed(42)
from collections import defaultdict

# from transformers.deepspeed import is_deepspeed_zero3_enabled
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control


LANG_TABLE = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Modern Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kirghiz",
    "li": "Limburgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "lo": "Laos",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovene",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wa": "Walloon",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}

TOKENIZER_LANG_TABLE = {
    "vi" : "vie_Latn",
    "km" : "khm_Khmr",
    "lo" : "lao_Laoo",
    "en" : "eng_Latn",
    "th" : "tha_Thai",
    "my" : "mya_Mymr",
    "zh" : "zho_Hans",
}

task_prompt = {
    "general_trans":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "doc_trans":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_medical":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_law":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_literature":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_colloquial":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_it":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "term_con_trans": [
        "Translate the following text from {src_lang} into {tgt_lang} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated.\nTerminology pairs: {term_text}\n{src_lang}: {src}"
    ],
    "ape": [
        "Improve the following machine-generated translation from {src_lang} to {tgt_lang}. Correct errors and generate a more accurate translation.\n{src_lang}: {src}\n{tgt_lang}: {mt_text}"
    ]
}


def check_add_eos(tokenized_inputs, tokenizer):
    if tokenized_inputs.input_ids[0][-1] != tokenizer.eos_token_id:
        for idx in range(len(tokenized_inputs.input_ids)):
            tokenized_inputs.input_ids[idx].append(tokenizer.eos_token_id)
            tokenized_inputs.attention_mask[idx].append(1)

def print_dataset(train_raw_data, valid_raw_data, test_raw_data):
    for part, part_data in  {"train":train_raw_data, "validation":valid_raw_data, "test":test_raw_data}.items():
        for lp, datas in part_data.items():
            for task, data in datas.items():
                print(f"{part}, {lp}, {task}, {len(data[part])}") 


def load_mmt_dataset(pairs, trans_task, data_args, model_args, training_args, logger):
    seen_files =set()
    train_raw_data, valid_raw_data, test_raw_data = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for pair in pairs:
        src_lang = pair.split("-")[0]
        tgt_lang = pair.split("-")[1]
        
        # first_lang = src_lang if src_lang != "en" else tgt_lang
        # second_lang = "en"
        first_lang = src_lang
        second_lang = tgt_lang
        pair_dir = f"{first_lang}-{second_lang}"
            
        for task in trans_task:
            train_file = os.path.join(data_args.mmt_data_path, pair_dir, f"train.{pair_dir}.{task}.json")
            valid_file = os.path.join(data_args.mmt_data_path, pair_dir, f"valid.{pair_dir}.{task}.json")

            # general_trans task may have multi test dataset
            if task == "general_trans":
                if data_args.test_dataname == "wmt23":
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}.wmt23*json"))
                elif data_args.test_dataname == "wmt22":
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}.wmt22*json"))
                elif data_args.test_dataname == "flores":
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}.FLORES-200*json"))
                else:
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}*json"))
            else:
                test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}*json"))

            if test_files:
                test_file = test_files[0]
            else:
                # fake file for logger
                test_file = f"test.{pair}.{task}.json"
            
            if not os.path.isfile(train_file):
                logger.info(f"Warning: training file {train_file} does not exist!")
            elif train_file not in seen_files and training_args.do_train:
                logger.info(f"Load training file {train_file}!")
                train_raw_data[f"{first_lang}-{second_lang}"][task] = load_dataset(
                    "json",
                    data_files={"train": train_file},
                    cache_dir=model_args.cache_dir,
                    # use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                    num_proc=training_args.dataloader_num_workers
                    )
            
            if not os.path.isfile(valid_file):
                logger.info(f"Warning: validation file {valid_file} does not exist!")
            elif valid_file not in seen_files and training_args.do_eval:
                logger.info(f"Load valid file {valid_file}!")
                valid_raw_data[f"{first_lang}-{second_lang}"][task] = load_dataset(
                    "json",
                    data_files={"validation": valid_file},
                    cache_dir=model_args.cache_dir,
                    # use_auth_token=True if model_args.use_auth_token else None,
                    num_proc=training_args.dataloader_num_workers
                    )
            
            if not os.path.isfile(test_file):
                logger.info(f"Warning: test file {test_file} does not exist!")
            elif test_file not in seen_files and training_args.do_predict:
                logger.info(f"Load test file {test_file}!")
                if data_args.override_test_data_path:
                    test_raw_data[f"{src_lang}-{tgt_lang}"][task] = load_dataset(
                        data_args.override_test_data_path,
                        f"{src_lang}-{tgt_lang}",
                        cache_dir=model_args.cache_dir,
                        # use_auth_token=True if model_args.use_auth_token else None,
                    )
                else:
                    test_raw_data[f"{src_lang}-{tgt_lang}"][task] = load_dataset(
                        "json",
                        data_files={"test": test_file},
                        cache_dir=model_args.cache_dir,
                        # use_auth_token=True if model_args.use_auth_token else None,
                    )

            seen_files.add(train_file)
            seen_files.add(valid_file)
            seen_files.add(test_file)
    print_dataset(train_raw_data, valid_raw_data, test_raw_data)
    return train_raw_data, valid_raw_data, test_raw_data


def get_prompt(source_lang, target_lang, example):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    task_type = example["task_type"]
    
    if task_type != "context_learning_trans":
        prefix_temp = random.choice(task_prompt[task_type])
    
    if task_type == "doc_trans":
        src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
    elif task_type == "term_con_trans":
        src_text, tgt_txt, hints = example["translation"][source_lang], example["translation"][target_lang], example["hints"]
        hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
        hint_text = " ; ".join(hints)
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, term_text=hint_text)
    elif task_type == "ape":
        src_text, tgt_txt, mt_text = example["translation"][source_lang], example["translation"][target_lang], example["mt_gen"]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, mt_text=mt_text)
    elif task_type == "context_learning_trans": 
        meta_task = example["meta_task"]
        shots = example["shots"]
        if meta_task == "term_con_trans":
            context = f"Translate the following text from {src_fullname} into {tgt_fullname} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated.\n"
            for shot in shots:
                src_text, tgt_txt, hints = shot["translation"][source_lang], shot["translation"][target_lang], shot["hints"]
                hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
                hint_text = " ; ".join(hints)
                context += f"Terminology pairs: {hint_text}\n{src_fullname}: {src_text}\n{tgt_fullname}: {tgt_txt}\n\n"
            src_text, tgt_txt, hints = example["translation"][source_lang], example["translation"][target_lang], example["hints"]
            hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
            hint_text = "; ".join(hints)
            prefix = context +  f"Terminology pairs: {hint_text}\n{src_fullname}: {src_text}"
        elif meta_task == "ape":
            context = f"Improve the following machine-generated translation from {src_fullname} to {tgt_fullname}. Correct errors and generate a more accurate translation.\n"
            for shot in shots:
                src_text, tgt_txt, mt_text = shot["translation"][source_lang], shot["translation"][target_lang], shot["mt_gen"]
                context += f"{src_fullname}: {src_text}\nMachine translation: {mt_text}\nImproved translation: {tgt_txt}\n\n"
            src_text, tgt_txt, mt_text = example["translation"][source_lang], example["translation"][target_lang], example["mt_gen"]
            prefix = context +  f"{src_fullname}: {src_text}\nMachine translation: {mt_text}"
        else:
            context = f"Translate the following text from {src_fullname} into {tgt_fullname}.\n"
            for shot in shots:
                src_text, tgt_txt = shot["translation"][source_lang], shot["translation"][target_lang]
                context += f"{src_fullname}: {src_text}\n{tgt_fullname}: {tgt_txt}\n\n"
            src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
            prefix = context + f"{src_fullname}: {src_text}"
    else:
        src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
    
    if task_type == "ape" or (task_type == "context_learning_trans" and meta_task == "ape"):
        suffix = "\nImproved translation: "
    else:
        suffix = f"\n{tgt_fullname}: "
    prompt = prefix + suffix
    return prompt, tgt_txt



def clean_outputstring(output, key_word, logger, split_idx):
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logger.info(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return out[1].strip()
        else:
            logger.info(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return out[2].strip()
    except:
        logger.info(f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")
        
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        logger.info(f"Can not solve the edge case, recover the translation to empty string! The output is {output}")
        return ""





def do_data_reverse(pairs, example):
    directional_tasks = ["ape"]
    directional_data_names = ["wmt19_robustness", "wmt20_robustness"]
    source_lang, target_lang, task_type, data_name = example["src_lang"], example["tgt_lang"], example["task_type"],  example["data_name"]
    flag = True
    if f"{target_lang}-{source_lang}" not in pairs or task_type in directional_tasks:
        flag = False
    # exclude general_trans data
    if task_type == "general_trans" and data_name in directional_data_names:
        flag = False
    # exclude some special fewshot task
    if task_type == "context_learning_trans" and example["meta_task"] in directional_data_names:
        flag = False
    return flag


def process_mmt_data_for_seq2seq(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, data_args, training_args):

    def tokenize_train_eval_for_seq2seq(examples):
        inputs, targets = [], []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:                
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                inputs.append(prompt)
                targets.append(tgt_txt)
            # if do_data_reverse(pairs, example):
            #     prompt, tgt_txt = get_prompt(target_lang, source_lang, example)
            #     inputs.append(prompt)
            #     targets.append(tgt_txt)
        # print(("\n\n"+"="*100+"\n\n").join([f"{x}\n{y}" for x,y in zip(inputs, targets)]))
        
        # add_special_tokens is not matter for the source
        formatted_inputs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in inputs
        ]

        model_inputs = tokenizer(
            formatted_inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        # model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=False, truncation=True)
        labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=False, truncation=True)
        # check_add_eos(labels, tokenizer)
        model_inputs["labels"] = labels["input_ids"]
    
        return model_inputs

    def tokenize_test_for_seq2seq(examples):
        prompts = []
        targets = []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                prompts.append(prompt)
                targets.append(tgt_txt)
        formatted_inputs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in inputs
        ]

        model_inputs = tokenizer(
            formatted_inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
    
        return model_inputs

    train_datasets, eval_datasets, test_datasets = None, None, None
    
    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                processed_datasets.append(train_dataset)   
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_datasets[lg_pair] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets[lg_pair][task] = test_dataset
    
    return train_datasets, eval_datasets, test_datasets

def process_mmt_data_for_llm(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer,  data_args, training_args):

    def tokenize_train_eval_left_pad(examples):
        prompts, inputs = [], []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                prompts.append(prompt)
                inputs.append(prompt + tgt_txt)
            # exclude some special tasks and dataset
            if do_data_reverse(pairs, example):
                prompt, tgt_txt = get_prompt(target_lang, source_lang, example)
                prompts.append(prompt)
                inputs.append(prompt + tgt_txt)
        # print(("\n\n"+"="*100+"\n\n").join(inputs)) # check data
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens - 1, truncation=True, add_special_tokens=True)
        check_add_eos(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs["input_ids"])

        if data_args.ignore_prompt_token_for_loss:
            for idx, prompt in enumerate(prompts):
                prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                labels[idx][: len(prompt)] = [-100] * len(prompt) 
        model_inputs["labels"] = labels
        return model_inputs
  
    def tokenize_test(examples):
        prompts, targets = [], []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                prompts.append(prompt)
                targets.append(prompt + tgt_txt)
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length,  truncation=True, add_special_tokens=False)
        
        return model_inputs

    train_datasets, eval_datasets, test_datasets = None, None, None

    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_left_pad,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                    processed_datasets.append(train_dataset)

        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_left_pad,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_datasets[lg_pair] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets[lg_pair][task] = test_dataset
    
    return train_datasets, eval_datasets, test_datasets

def load_data_pretrain(languages, data_args, model_args, training_args, logger):
    seen_files =set()
    train_raw_data, valid_raw_data, test_raw_data = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for lang in languages:
        train_file = os.path.join(data_args.mmt_data_path, lang, f"train.{lang}.json")
        valid_file = os.path.join(data_args.mmt_data_path, lang, f"valid.{lang}.json")
        test_file = os.path.join(data_args.mmt_data_path, lang, f"test.{lang}.json")

        if not os.path.isfile(train_file):
            logger.info(f"Warning: training file {train_file} does not exist!")
        elif train_file not in seen_files and training_args.do_train:
            logger.info(f"Load training file {train_file}!")
            train_raw_data[lang]["pretrain"] = load_dataset(
                "json",
                data_files={"train": train_file},
                cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
                num_proc=training_args.dataloader_num_workers
                )
        
        if not os.path.isfile(valid_file):
            logger.info(f"Warning: validation file {valid_file} does not exist!")
        elif valid_file not in seen_files and training_args.do_eval:
            logger.info(f"Load valid file {valid_file}!")
            valid_raw_data[lang]["pretrain"] = load_dataset(
                "json",
                data_files={"validation": valid_file},
                cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
                num_proc=training_args.dataloader_num_workers
                )
        
        if not os.path.isfile(test_file):
            logger.info(f"Warning: test file {test_file} does not exist!")
        elif test_file not in seen_files and training_args.do_predict:
            logger.info(f"Load test file {test_file}!")
            test_raw_data[lang]["pretrain"] = load_dataset(
                "json",
                data_files={"test": test_file},
                cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
            )

        seen_files.add(train_file)
        seen_files.add(valid_file)
        seen_files.add(test_file)
    print_dataset(train_raw_data, valid_raw_data, test_raw_data)
    return train_raw_data, valid_raw_data, test_raw_data

def process_pretrain_data_for_seq2seq(train_raw_data, valid_raw_data, test_raw_data, languages, tokenizer, data_args, training_args):

    def tokenize_train_eval_for_seq2seq(examples):
        input_ids = []
        labels = []
        attention_masks = []

        examples = [
            {key: value for key, value in zip(examples.keys(), values)}
            for values in zip(*examples.values())
        ]

        for example in examples:

            lang = example["lang"]

            if lang in languages:

                text = example["text"]

                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=data_args.max_source_length + data_args.max_target_length,
                    padding=False
                )["input_ids"]

                if len(tokens) < 30:
                    continue

                # random split 30%–70%
                split = random.randint(
                    int(len(tokens) * 0.4),
                    int(len(tokens) * 0.6)
                )

                prefix = tokens[:split]
                target = tokens[split:]

                # thêm EOS cho target
                if target[-1] != tokenizer.eos_token_id:
                    target.append(tokenizer.eos_token_id)

                input_ids.append(prefix)
                labels.append(target)
                attention_masks.append([1] * len(prefix))
        # check_add_eos(labels, tokenizer)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }

        return model_inputs
    
    def tokenize_test_for_seq2seq(examples):
        input_ids = []
        attention_masks = []

        examples = [
            {key: value for key, value in zip(examples.keys(), values)}
            for values in zip(*examples.values())
        ]

        for example in examples:

            lang = example["lang"]

            if lang in languages:

                text = example["text"]

                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=data_args.max_source_length + data_args.max_target_length,
                    padding=False
                )["input_ids"]

                split = random.randint(
                    int(len(tokens) * 0.4),
                    int(len(tokens) * 0.6)
                )

                prefix = tokens[:split]

                input_ids.append(prefix)
                attention_masks.append([1] * len(prefix))

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_masks
        }

        return model_inputs
    
    train_datasets, eval_datasets, test_datasets = None, None, None
    if training_args.do_train:
        processed_datasets = []
        for lang, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                # print(f"Train datasets column names: {train_dataset.column_names}")
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on pretrain train dataset",
                    )
                processed_datasets.append(train_dataset)
                # print(f"Train datasets column names: {train_dataset.column_names}")
                
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        # print(f"Finish processing pretrain train dataset, the number of samples is {len(train_datasets)}")

    if training_args.do_eval:
        processed_datasets = []
        for lang, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on pretrain valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)


    if training_args.do_predict:
        test_datasets = {}
        for lang, sub_raw_data in test_raw_data.items():
            test_datasets[lang] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on pretrain test dataset",
                    )
                test_datasets[lang][task] = test_dataset
    # print(f"Train datasets column names: {train_datasets.column_names}")
    # print(f"Evaluation datasets column names: {eval_datasets.column_names}")
    # print(f"Test datasets column names: {test_datasets[lang][task].column_names}")
    return train_datasets, eval_datasets, test_datasets

def process_mmt_data_for_seq2seq_ver2(train_raw_data, valid_raw_data, test_raw_data, pairs, llm_tokenizer, seq2seq_tokenizer, data_args, training_args):

    def tokenize_train_eval_for_seq2seq(examples):
        inputs, targets, tgt_langs, inputs_2 = [], [], [], []

        examples = [
            {key: value for key, value in zip(examples.keys(), values)}
            for values in zip(*examples.values())
        ]

        for example in examples:
            src_lang, tgt_lang = example["src_lang"], example["tgt_lang"]

            if f"{src_lang}-{tgt_lang}" in pairs:
                prompt, tgt_txt = get_prompt(src_lang, tgt_lang, example)
                inputs.append(prompt)
                targets.append(tgt_txt)
                tgt_langs.append(tgt_lang)
                prompt_2, tgt_txt_2 = get_prompt(tgt_lang, tgt_lang, example)
                inputs_2.append(prompt_2)

            # if do_data_reverse(pairs, example):
            #     prompt, tgt_txt = get_prompt(tgt_lang, src_lang, example)
            #     inputs.append(prompt)
            #     targets.append(tgt_txt)
            #     tgt_langs.append(src_lang)
        for target in targets:
            if target == None:
                print("Detect None target, replace it with empty string!")
        formatted_inputs = [
            llm_tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in inputs
        ]

        model_inputs = llm_tokenizer(
            formatted_inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        model_inputs = dict(model_inputs)

        formatted_inputs_2 = [
            llm_tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in inputs_2
        ]

        model_inputs_2 = llm_tokenizer(
            formatted_inputs_2,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        model_inputs_2 = dict(model_inputs_2)

        labels = seq2seq_tokenizer(
            targets,
            max_length=data_args.max_target_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        labels = dict(labels)

        new_labels, new_masks = [], []

        for tgt_lang, ids, mask in zip(
            tgt_langs, labels["input_ids"], labels["attention_mask"]
        ):
            bos_id = seq2seq_tokenizer.convert_tokens_to_ids(
                TOKENIZER_LANG_TABLE[tgt_lang]
            )

            if ids[0] == bos_id:
                ids[0] = bos_id
                mask = [1] + mask

            new_labels.append(ids)
            new_masks.append(mask)

        # mask padding -> -100
        final_labels = []
        for ids, mask in zip(new_labels, new_masks):
            final_labels.append([
                token if m == 1 else -100
                for token, m in zip(ids, mask)
            ])

        model_inputs["labels"] = new_labels

        model_inputs["input_ids_2"] = model_inputs_2["input_ids"]
        model_inputs["attention_mask_2"] = model_inputs_2["attention_mask"]

        return model_inputs

    def tokenize_test_for_seq2seq(examples):
        prompts, targets, tgt_langs = [], [], []

        # HF batch -> list[dict]
        examples = [
            {key: value for key, value in zip(examples.keys(), values)}
            for values in zip(*examples.values())
        ]

        for example in examples:
            src_lang, tgt_lang = example["src_lang"], example["tgt_lang"]

            if f"{src_lang}-{tgt_lang}" in pairs:
                prompt, tgt_txt = get_prompt(src_lang, tgt_lang, example)
                prompts.append(prompt)
                targets.append(tgt_txt)
                tgt_langs.append(tgt_lang)

        # =========================
        # 1. TOKENIZE INPUT (LLM)
        # =========================
        formatted_inputs = [
            llm_tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in prompts
        ]

        model_inputs = llm_tokenizer(
            formatted_inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        model_inputs = dict(model_inputs)

        # =========================
        # 2. TOKENIZE TARGET (for evaluation only)
        # =========================
        # labels = seq2seq_tokenizer(
        #     text_target=targets,
        #     max_length=data_args.max_target_length,
        #     padding=False,
        #     truncation=True,
        # )

        # check_add_eos(labels, seq2seq_tokenizer)

        # # =========================
        # # 3. FORCE TARGET LANGUAGE (QUAN TRỌNG)
        # # =========================
        forced_bos_token_ids = []
        for lang in tgt_langs:
            bos_token_id = seq2seq_tokenizer.convert_tokens_to_ids(TOKENIZER_LANG_TABLE[lang])
            forced_bos_token_ids.append([bos_token_id])
        # forced_bos_token_ids = [
        #     seq2seq_tokenizer.convert_tokens_to_ids(TOKENIZER_LANG_TABLE[lang])
        #     for lang in tgt_langs
        # ]

        # model_inputs["labels"] = labels["input_ids"]  # dùng để tính metric
        model_inputs["decoder_input_ids"] = forced_bos_token_ids
        # print(f"Forced BOS token ids for evaluation: {forced_bos_token_ids}")
        # print(f"model_inputs[\"forced_bos_token_id\"]: {model_inputs['forced_bos_token_id']}")

        return model_inputs

    train_datasets, eval_datasets, test_datasets = None, None, None
    
    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                processed_datasets.append(train_dataset)   
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_datasets[lg_pair] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets[lg_pair][task] = test_dataset
    
    return train_datasets, eval_datasets, test_datasets
