import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict, Any

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers import AutoModelForCausalLM
from qwen.models.encoder import QwenModelEncoder
from qwen.models.modules.connector import Connector, GroupedEncoderFusion
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel,
)


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

## Qwen + Connector(MLP + light Encoder)
# class QwenModelCombineEncoder(QwenModelEncoder):
#     def __init__(self, config: Qwen2Config):
#         super().__init__(config)
#         self.connector = Connector(config)
#         self.fuse_model = GroupedEncoderFusion(config, group_size=4)

#     @add_start_docstrings_to_model_forward(Qwen_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
        
#         encoder_outputs = super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=False,
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=return_dict,
#         )

#         output_hidden_states = encoder_outputs.hidden_states
#         hidden_states = encoder_outputs.hidden_states[1:]
#         # print(type(hidden_states))
#         fuse_hidden_state = self.fuse_model(output_hidden_states[1:])  # exclude embedding
#         last_hidden_state = fuse_hidden_state
#         hidden_states = hidden_states + (fuse_hidden_state,)

#         # last_hidden_state = encoder_outputs.last_hidden_state
        
#         connector_outputs = self.connector(
#             last_hidden_state,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         hidden_states = hidden_states + connector_outputs.hidden_states

#         return BaseModelOutputWithPast(
#             last_hidden_state=connector_outputs.last_hidden_state,
#             past_key_values=None,
#             hidden_states=hidden_states, # for compatible other's decoder
#             attentions=None,
#         )


# class QwenModelCombineEncoder(nn.Module):
#     def __init__(
#         self,
#         config: Qwen2Config,
#         model_name_or_path: Optional[str] = None,
#     ):
#         super().__init__()

#         if model_name_or_path is None:
#             # Chỉ khởi tạo cấu trúc từ config, chưa nạp pretrained weights
#             causal_lm = AutoModelForCausalLM.from_config(config)
#         else:
#             # Nạp mô hình pretrained
#             causal_lm = AutoModelForCausalLM.from_pretrained(
#                 model_name_or_path,
#                 config=config,
#                 torch_dtype="auto",
#             )

#         # Chỉ giữ phần Qwen2Model, không dùng lm_head
#         self.llm = causal_lm

#         self.connector = Connector(config)
#         self.fuse_model = GroupedEncoderFusion(
#             config,
#             group_size=4,
#         )

#     @add_start_docstrings_to_model_forward(Qwen_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:

#         # Phần sau cần truy cập .hidden_states,
#         # vì vậy nên buộc return_dict=True
#         encoder_outputs = self.llm.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=None,
#             inputs_embeds=inputs_embeds,
#             use_cache=False,
#             output_attentions=output_attentions,
#             output_hidden_states=True,
#             return_dict=True,
#             cache_position=cache_position,
#         )

#         # Tuple gồm:
#         # hidden_states[0]: embedding output
#         # hidden_states[1:]: output của các Transformer layer
#         llm_hidden_states = encoder_outputs.hidden_states
#         transformer_hidden_states = llm_hidden_states[1:]

#         fuse_hidden_state = self.fuse_model(
#             transformer_hidden_states
#         )

#         all_hidden_states = transformer_hidden_states + (
#             fuse_hidden_state,
#         )

#         connector_outputs = self.connector(
#             fuse_hidden_state,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )

#         if connector_outputs.hidden_states is not None:
#             all_hidden_states = (
#                 all_hidden_states
#                 + tuple(connector_outputs.hidden_states)
#             )

#         outputs = BaseModelOutputWithPast(
#             last_hidden_state=connector_outputs.last_hidden_state,
#             past_key_values=None,
#             hidden_states=all_hidden_states,
#             attentions=encoder_outputs.attentions,
#         )

#         if return_dict is False:
#             return outputs.to_tuple()

#         return outputs



import torch

from typing import List, Optional, Tuple, Union

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Qwen2Config,
    AutoConfig
)

from transformers.modeling_outputs import BaseModelOutputWithPast


class QwenModelCombineEncoder(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "llm"

    def __init__(self, config: Qwen2Config):
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        super().__init__(config)

        # Có cả Qwen backbone và lm_head
        llm_config = AutoConfig.from_pretrained(config.llm_path, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_config(llm_config)

        # print(config, type(config), config.decoder)

        self.connector = Connector(config)
        self.fuse_model = GroupedEncoderFusion(
            config,
            group_size=4,
        )

        self.post_init()

    @add_start_docstrings_to_model_forward(Qwen_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        # CausalLMOutputWithPast.hidden_states:
        # [0]       embedding output
        # [1:]      output của các Transformer layer
        llm_hidden_states = llm_outputs.hidden_states[1:]

        fuse_hidden_state = self.fuse_model(
            llm_hidden_states
        )

        hidden_states = llm_hidden_states + (
            fuse_hidden_state,
        )

        connector_outputs = self.connector(
            fuse_hidden_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if connector_outputs.hidden_states is not None:
            hidden_states = (
                hidden_states
                + tuple(connector_outputs.hidden_states)
            )

        outputs = BaseModelOutputWithPast(
            last_hidden_state=connector_outputs.last_hidden_state,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=llm_outputs.attentions,
        )

        if return_dict is False:
            return outputs.to_tuple()

        return outputs