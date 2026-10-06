import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    llm_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    mt_model_path: str = field(
        default=None, metadata={"help": "Path to pretrained decoder model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


    ## new add
    model_method: str = field(
        default="default", 
        metadata={
            "help": "The default refers to the general seq2seq model, such as t5,bart"
        },
    )
    run_mode: str = field(default="resume")
    do_sample: bool = field(default=False)
    patience: int = field(default=3)

    model_method: str = field(default="SailorED", metadata={"help": "The method to build the model architecture"})

    num_connector_layers: int = field(default=4),
    connector_hidden_size: int = field(default=1024),
    connector_intermediate_size: int = field(default=4096),
    connector_num_attention_heads: int = field(default=16),
    connector_num_key_value_heads: int = field(default=8),
    connector_model_method: str = field(default="stack"),
    fuse_model_group_size: int = field(default=4),

    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    contrastive_lambda: float = field(default=0.0)
    contrastive_temperature: float = field(default=0.1)

    ot_lambda: float = field(default=0.0)
    ot_reg: float = field(default=0.1)
    ot_num_iters: int = field(default=20)
    ot_eps: float = field(default=1e-8)

    freeze_llm: bool = field(default=True)
    freeze_decoder: bool = field(default=True)
    freeze_decoder_cross_attn: bool = field(default=True) 
    freeze_mt_lm_head: bool = field(default=True)
    train_lora_llm: bool = field(default=False) 
    train_lora_mt: bool = field(default=False)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    override_test_data_path: Optional[str] = field(default=None, metadata={"help": "This will override the default test data in the mmt data"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum new tokens to generate except the prompt."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )
    right_pad: bool = field(
        default=False,
        metadata={
            "help": "Use right pad for training, especially for models like MPT."
        },
    )

    ### add new
    language_pairs: str = field(default="de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh,vi-km,vi-lo", metadata={"help": "training language pairs"})
    mmt_data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
 
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of test examples to this "
                "value if set."
            )
        },
    )

    trans_task: str = field(
        default="general_trans"
    )
    predict_task: str = field(
        default="general_trans"
    )
    test_dataname: str = field(
        default="wmt23"
    )

    languages: str = field(
        default="de,en,cs,ru,zh,vi,km,lo"
    )



    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length