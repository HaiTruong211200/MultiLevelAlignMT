import os
import random
random.seed(42)
import torch
import regex
import glob

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
)
# from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerCallback
from safetensors import safe_open

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



def is_whitespace(string):
    # 使用正则表达式匹配空白字符或不可见字符
    pattern = r'^[\s\p{C}[\x00-\xFF]]+$'
    match = regex.match(pattern, string)
    return match is not None


def load_checkpoint(model_path):
    ## load checkpoint
    checkpoint_url = glob.glob(f"{model_path}/*model.bin")
    state = {}
    if len(checkpoint_url) != 0:
        for part in checkpoint_url:
            state.update(torch.load(part))
    else:
        checkpoint_url = glob.glob(f"{model_path}/*safetensors")
        if len(checkpoint_url) == 0:
            print("No checkpoint!")
            exit()
        for part in checkpoint_url:
            with safe_open(part, framework="pt") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
    return state

from transformers import AutoModelForCausalLM

def make_model_state_dict(model_path, seq2seq_model_name_or_path):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    state = model.state_dict()

    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(
        seq2seq_model_name_or_path,
        trust_remote_code=True
    )

    seq2seq_state = seq2seq_model.state_dict()

    new_state = {}
    for key, value in state.items():
        if key.startswith("model"):
            key = "encoder" + key[5:]
        new_state[key] = value

    for key, value in seq2seq_state.items():
        new_key = "mt_model." + key
        new_state[new_key] = value
        # if key.startswith("model"):
        #     new_key = "decoder.mt_model.model.decoder" + key[5:]
        #     new_state[new_key] = value
        # elif key.startswith("model.lm_head"):
        #     new_key = "lm_head" + key[13:]
        #     new_state[new_key] = value

    llm_model = AutoModelForCausalLM.from_pretrained(
        "hai2131/SailorED-Stage2-672"
    )

    llm_state = llm_model.state_dict()

    for key, value in llm_state.items():
        new_key = "llm." + key
        new_state[new_key] = value


    return new_state

# def make_model_state_dict_ver2(model_path):   
#     ## get encoder state and lm_head 
#     state = load_checkpoint(model_path)
#     new_state = {}
#     for key, value in state.items():
#         if key.startswith("model"):
#             key =  "encoder" + key[5:]
#         # lm_head
#         new_state[key] = value
#     return new_state

def set_model_special_tokens(model, model_name_or_path):
    if "Llama-2" in model_name_or_path or "Tower" in model_name_or_path or "ALMA" in model_name_or_path:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
    elif "Qwen1.5" in model_name_or_path or "Qwen2" in model_name_or_path:
        model.config.pad_token_id = 151644
        model.config.bos_token_id = 151643
        model.config.eos_token_id = 151643
        model.generation_config.pad_token_id = 151644
        model.generation_config.bos_token_id = 151643
        model.generation_config.eos_token_id = 151643
    elif "Llama-3" in model_name_or_path:
        model.config.pad_token_id = 128002
        model.generation_config.pad_token_id = 128002
    elif "Qwen3" in model_name_or_path or "Sailor" in model_name_or_path or "qwen" in model_name_or_path:
        model.config.pad_token_id = 151644
        model.config.bos_token_id = 151643
        model.config.eos_token_id = 151643
        model.generation_config.pad_token_id = 151644
        model.generation_config.bos_token_id = 151643
        model.generation_config.eos_token_id = 151643
    elif "Qwen2.5" in model_name_or_path :
        model.config.pad_token_id = 151644
        model.config.bos_token_id = 151643
        model.config.eos_token_id = 151643
        model.generation_config.pad_token_id = 151644
        model.generation_config.bos_token_id = 151643
        model.generation_config.eos_token_id = 151643
    return model

def set_tokenizer_special_tokens(tokenizer, model_name_or_path):
    if "Llama-2" in model_name_or_path or "Tower" in model_name_or_path or "ALMA" in model_name_or_path:
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"
        tokenizer.bos_token = "<s>"
    elif "Qwen1.5" in model_name_or_path or "Qwen2" in model_name_or_path:
        tokenizer.pad_token_id = 151644
        tokenizer.bos_token_id = 151643
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token = "<|im_start|>"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
    elif "Llama-3" in model_name_or_path:
        tokenizer.pad_token_id = 128002
    elif "Qwen3" in model_name_or_path or "Sailor" in model_name_or_path or "qwen" in model_name_or_path:
        tokenizer.pad_token_id = 151644
        tokenizer.bos_token_id = 151643
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token = "<|im_start|>"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
    elif "Qwen2.5" in model_name_or_path:
        tokenizer.pad_token_id = 151644
        tokenizer.bos_token_id = 151643
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token = "<|im_start|>"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
    return tokenizer


def load_model(data_args, model_args, training_args, tokenizer, logger):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "max_length": data_args.max_source_length + data_args.max_new_tokens,
        # "norm_type": "low_precision_rmsnorm",
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    ## Model Loading
    if model_args.model_name_or_path:
        model =  AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path ,
            config=config,
        )

        model.generation_config.max_length = data_args.max_source_length + data_args.max_new_tokens
        model.generation_config.use_cache = True
        # when do inference only
        if not training_args.do_train and config.torch_dtype is torch.float32:
            model = model.half()
            logger.info("Model dtype is torch.float32, chanege to torch.float16 for inference only")

    ## train from scratch
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = set_model_special_tokens(model, model_args.llm_path)
    logger.info(model)
    return model


def load_tokenizer(data_args, model_args, training_args, logger, add_eos_token=False):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "padding_side": 'left' if not data_args.right_pad else "right",
        "add_eos_token": add_eos_token,
        "trust_remote_code": True
    }
        
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.llm_path:
        if "llama" in model_args.llm_path or "ALMA" in model_args.llm_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_args.llm_path, 
                **tokenizer_kwargs, 
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.llm_path,
                **tokenizer_kwargs,
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer = set_tokenizer_special_tokens(tokenizer, model_args.llm_path)
    return tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_train_module(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.cuda.current_device() == 0:
                print(f"train  ==> {name}")
