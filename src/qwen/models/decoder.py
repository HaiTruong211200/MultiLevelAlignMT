import torch
from torch import nn
from typing import List, Optional, Tuple, Union, Dict, Any

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers import AutoModelForSeq2SeqLM

from qwen.models.modules.normalization import QwenRMSNorm
from .base_model import QwenPreTrainedModel
from qwen.models.modules.layers import QwenCrossAttDecoderLayer


logger = logging.get_logger(__name__)

Qwen_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""



class QwenCrossAttDecoder(QwenPreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QwenCrossAttDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
               
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(Qwen_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,        
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_all_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_attention_mask:  Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            # if not isinstance(past_key_values, StaticCache):
            if past_key_values is None:
                # self-attention, cross-attention
                past_key_values = (DynamicCache.from_legacy_cache(past_key_values), DynamicCache.from_legacy_cache(past_key_values))
                past_seen_tokens = past_key_values[0].get_seq_length()
            else:
                past_key_values = (DynamicCache.from_legacy_cache(past_key_values[0]), DynamicCache.from_legacy_cache(past_key_values[1]))
                past_seen_tokens = past_key_values[0].get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values[0], StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            # cache_position = torch.arange(
            #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            # )

            ## new
            cache_position = torch.arange(past_seen_tokens, attention_mask.size(1), device=inputs_embeds.device)

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # do encoder position ids and attention mask on-the-fly
        encoder_position_ids = encoder_attention_mask.long().cumsum(-1) - 1
        encoder_position_ids.masked_fill_(encoder_attention_mask == 0, 1)
        encoder_mask = self._update_causal_mask(encoder_attention_mask, inputs_embeds, cache_position, past_seen_tokens, is_biderectional=True)

        # embed positions
        hidden_states = inputs_embeds
            
        last_encoder_hidden = encoder_all_hidden_states[-1]
        fuse_encoder_hidden_states = [last_encoder_hidden for _ in range(len(self.layers))]
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i,decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    encoder_hidden_states=fuse_encoder_hidden_states[i],
                    encoder_attention_mask=encoder_mask,
                    encoder_position_ids=encoder_position_ids
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = tuple(
                (cur_cache.to_legacy_cache() if isinstance(cur_cache, Cache) else cur_cache) for cur_cache in next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_seen_tokens, is_biderectional=False):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min

        sequence_length = input_tensor.shape[1]
        
        if hasattr(self.layers[0].self_attn, "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] 
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        
        ## bidirectional attention 
        if is_biderectional:
            causal_mask = torch.full((sequence_length, target_length), fill_value=0., dtype=dtype, device=device)
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        
        ## get triu matrix and then multiply to get future mask
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
class NLLBDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # full NLLB model
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path,
            device_map=None,
        )

        # shortcuts
        # self.model = self.mt_model.model

        # self.encoder = self.model.encoder
        # self.decoder = self.model.decoder

        # self.shared = self.model.shared
        # # self.lm_head = self.mt_model.lm_head

        # # decoder shortcuts
        # self.embed_tokens = self.decoder.embed_tokens
        # self.embed_positions = self.decoder.embed_positions
        # self.layers = self.decoder.layers
        # self.layer_norm = self.decoder.layer_norm

        self.vocab_size = self.mt_model.config.vocab_size

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,        
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_all_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_attention_mask:  Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.mt_model.model.decoder.embed_tokens(input_ids) + self.mt_model.model.decoder.embed_positions(position_ids)
        
        
        hidden_states = inputs_embeds

        # for i,decoder_layer in enumerate(self.layers):
        #     hidden_states = decoder_layer(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         # position_ids=position_ids,
        #         past_key_value=past_key_values,
        #         output_attentions=output_attentions,
        #         use_cache=use_cache,
        #         cache_position=cache_position,
        #         encoder_hidden_states=encoder_all_hidden_states[-1],
        #         encoder_attention_mask=encoder_attention_mask
        #     )[0]

        # hidden_states = self.norm(hidden_states)
        outputs = self.mt_model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_all_hidden_states[-1],
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs