import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import inspect

from dataclasses import dataclass

from transformers import  PretrainedConfig, AutoConfig, GenerationConfig
from transformers import GenerationMixin
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.modeling_outputs import Seq2SeqLMOutput,  ModelOutput, BaseModelOutput
from transformers.utils import is_accelerate_available
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .base_model import QwenPreTrainedModel
from .encoder import QwenModelEncoder
from .combine_encoder import QwenModelCombineEncoder
from .decoder import QwenCrossAttDecoder, NLLBDecoder
from .modules.connector import Connector, GroupedEncoderFusion

def print_train_module(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.cuda.current_device() == 0:
                print(f"train  ==> {name}")

@dataclass
class CustomSeq2SeqLMOutput(Seq2SeqLMOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    ot_loss: Optional[torch.FloatTensor] = None
class QwenForSeq2SeqConfig(Qwen2Config):
    def __init__(
        self, 
        mt_model_path = None, 
        llm_path = None,
        num_connector_layers= None,
        connector_hidden_size = None,
        connector_intermediate_size = None,
        connector_num_attention_heads = None,
        connector_num_key_value_heads = None,
        connector_model_method = None,
        fuse_model_group_size = None,
        contrastive_lambda = 0.0,
        contrastive_temperature = 0.07,
        ot_lambda = 0.0,
        ot_reg = 0.1,
        ot_num_iters = 20,
        ot_eps = 1e-8,
        **kwargs):
        super().__init__(**kwargs)
        self.mt_model_path = mt_model_path
        self.llm_path = llm_path
        self.num_connector_layers = num_connector_layers
        self.connector_hidden_size = connector_hidden_size
        self.connector_intermediate_size = connector_intermediate_size
        self.connector_num_attention_heads = connector_num_attention_heads
        self.connector_num_key_value_heads = connector_num_key_value_heads
        self.connector_model_method = connector_model_method
        self.fuse_model_group_size = fuse_model_group_size
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temperature
        self.ot_lambda = ot_lambda
        self.ot_reg = ot_reg
        self.ot_num_iters = ot_num_iters
        self.ot_eps = ot_eps

class QwenModelForSeq2Seq(QwenPreTrainedModel):
    def __init__(self, config: QwenForSeq2SeqConfig, is_init=False):
        super().__init__(config)

        self.llm_config = AutoConfig.from_pretrained(config.llm_path)
        self.mt_config = AutoConfig.from_pretrained(config.mt_model_path)

        if is_init:
            self.llm = AutoModelForCausalLM.from_pretrained(config.llm_path)
            self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(config.mt_model_path)
        else:
            self.llm = AutoModelForCausalLM.from_config(config=self.llm_config)
            self.mt_model = AutoModelForSeq2SeqLM.from_config(config=self.mt_config)
        adapter_config = copy.deepcopy(config.to_dict())
        adapter_config["num_hidden_layers"] = config.num_connector_layers
        adapter_config["hidden_size"] = config.connector_hidden_size
        adapter_config["intermediate_size"] = config.connector_intermediate_size
        adapter_config["num_attention_heads"] = config.connector_num_attention_heads
        adapter_config["num_key_value_heads"] = config.connector_num_key_value_heads
        adapter_config["encoder_method"] = config.connector_model_method
        adapter_config["layer_types"] = ["full_attention"] * config.num_connector_layers
        adapter_config["llm_hidden_size"] = self.llm_config.hidden_size
        adapter_config = Qwen2Config(**adapter_config)

        self.adapter_config = adapter_config
        
        self.connector = Connector(self.adapter_config)
        self.fuse_model = GroupedEncoderFusion(self.llm_config, config.fuse_model_group_size)
        

        self.contrastive_lambda = config.contrastive_lambda
        self.contrastive_temperature = config.contrastive_temperature
        self.ot_lambda = config.ot_lambda
        self.ot_reg = config.ot_reg
        self.ot_num_iters = config.ot_num_iters
        self.ot_eps = config.ot_eps


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_2: Optional[torch.LongTensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CustomSeq2SeqLMOutput]:
        # if inputs_embeds is None:
        #     inputs_embeds = self.llm.embed_tokens(input_ids)
        # hidden_states = inputs_embeds
        # all_hidden_states = ()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if input_ids_2 is None:
        #     raise ValueError("input_ids_2 is required when contrastive_lambda > 0 or ot_lambda > 0")
        # else:
        #     print(f"input_ids_2 shape: {input_ids_2.shape}, attention_mask_2 shape: {attention_mask_2.shape}")
        if labels is not None:
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            

            encoder_hidden_states = llm_outputs.hidden_states[1:]
            fuse_hidden_state = self.fuse_model(encoder_hidden_states)  # exclude embedding
            last_hidden_state = fuse_hidden_state
            encoder_hidden_states = encoder_hidden_states + (fuse_hidden_state,)

            # last_hidden_state = encoder_outputs.last_hidden_state
            
            connector_outputs = self.connector(
                last_hidden_state,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            encoder_hidden_states = encoder_hidden_states + connector_outputs.hidden_states

            if self.contrastive_lambda > 0 or self.ot_lambda > 0:
                encoder_hidden_states_2 = self.llm(
                    input_ids=input_ids_2,
                    attention_mask=attention_mask_2,
                    inputs_embeds=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[1:]

                fuse_hidden_state_2 = self.fuse_model(encoder_hidden_states_2)  # exclude embedding
                last_hidden_state_2 = fuse_hidden_state_2
                encoder_hidden_states_2 = encoder_hidden_states_2 + (fuse_hidden_state_2,)

                connector_outputs_2 = self.connector(
                    last_hidden_state_2,
                    input_ids=input_ids_2,
                    attention_mask=attention_mask_2,
                )
                encoder_hidden_states_2 = encoder_hidden_states_2 + connector_outputs_2.hidden_states

            decoder_outputs = self.mt_model.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_hidden_states[-1],
                encoder_attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                    # position_ids=position_ids,
            )
                # print(decoder_outputs.keys())
            decoder_hidden_states = decoder_outputs[0]
            ## compute loss 
            pretraining_tp = getattr(self.config, "pretraining_tp", 1)
            if pretraining_tp is not None and pretraining_tp > 1:
                print(f"Using pretraining_tp={pretraining_tp} for parallel LM head computation!")
                lm_head_slices = self.mt_model.lm_head.weight.split(self.vocab_size // pretraining_tp, dim=0)
                logits = [F.linear(decoder_hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.mt_model.lm_head(decoder_hidden_states)
            # print(f"Logits shape: {logits.shape}, labels shape: {labels.shape}")
            logits = logits.float()
            
            loss = None
            if labels is not None:
                # print(f"labels shape: {labels.shape}, logits shape: {logits.shape}")
                loss_fct = CrossEntropyLoss()
                labels = labels.to(logits.device)
                lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = lm_loss
                if self.contrastive_lambda > 0:
                    contrastive_loss = self.compute_contrastive_loss(encoder_hidden_states[-1],  encoder_hidden_states_2[-1])
                    # contrastive_loss = self.compute_contrastive_loss(encoder_all_hidden_states[-1], dec_hidden_states)
                    loss = lm_loss + self.contrastive_lambda * contrastive_loss
                if self.ot_lambda > 0:
                    ot_loss = self.compute_ot_loss_cosine(
                        hidden_states_a=encoder_hidden_states[-1],
                        mask_a=attention_mask,
                        hidden_states_b=encoder_hidden_states_2[-1],
                        mask_b=attention_mask_2,
                        reg=self.ot_reg,
                        num_iters=self.ot_num_iters,
                        eps=self.ot_eps
                    )
                    # print(f"OT Loss: {ot_loss.item():.4f}")
                    loss = loss + self.ot_lambda * ot_loss

            if not return_dict:
                    output = (logits,) + decoder_outputs[1:]
                    return (loss,) + output if loss is not None else output
            
            return CustomSeq2SeqLMOutput(
                loss=loss,
                lm_loss=lm_loss.detach(),
                contrastive_loss=contrastive_loss.detach() if self.contrastive_lambda > 0 else None,
                ot_loss=ot_loss.detach() if self.ot_lambda > 0 else None,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                # decoder_hidden_states=decoder_outputs,
                cross_attentions=None,
                encoder_last_hidden_state=encoder_hidden_states[-1],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attentions=None,
            )
        else:
            
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            output_hidden_states = llm_outputs.hidden_states
            fuse_hidden_state = self.fuse_model(output_hidden_states[1:])  # exclude embedding
            last_hidden_state = fuse_hidden_state

            # last_hidden_state = encoder_outputs.last_hidden_state
            
            connector_outputs = self.connector(
                last_hidden_state,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = connector_outputs.last_hidden_state
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=last_hidden_states,
            )
            
            
            # decoder_generate_ids_list = []

            generation_config = GenerationConfig(
                forced_bos_token_id=decoder_input_ids[0, 0].item(),
                max_new_tokens=200,
                num_beams=5
            )
            

            decoder_generate_ids = self.mt_model.generate(
                input_ids=decoder_input_ids,
                # forced_bos_token_id=decoder_input_ids[0, 0].item(),
                generation_config=generation_config,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask
            )

            # decoder_generate_ids_list.append(decoder_generate_ids)

            if len(decoder_generate_ids) == 1:
                return (input_ids, decoder_generate_ids[0])
            else:
                return (input_ids, decoder_generate_ids)
    
    def compute_contrastive_loss(self, encoder_hidden_states, decoder_hidden_states):
        # encoder_hidden_states: (batch_size, seq_len, hidden_size)
        # decoder_hidden_states: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        anchor_features = encoder_hidden_states.sum(dim=1) / encoder_hidden_states.size(1)  # (batch_size, hidden_size)
        positive_features = decoder_hidden_states.sum(dim=1) / decoder_hidden_states.size(1)  # (batch_size, hidden_size)

        npairs, hidden_size = anchor_features.shape

        similarity_function = nn.CosineSimilarity(dim=-1)

        anchor_dot_contrast = similarity_function(anchor_features.expand((npairs, npairs, hidden_size)),
                                                   positive_features.expand((npairs, npairs, hidden_size)).transpose(0, 1))  # (batch_size, batch_size)
        
        contrastive_loss_i = -nn.LogSoftmax(dim=0)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum() # (batch_size,)

        contrastive_loss_j = -nn.LogSoftmax(dim=1)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum() # (batch_size,)
        return (contrastive_loss_i + contrastive_loss_j) / 2.0 / npairs


    def compute_ot_loss_cosine(
        self,
        hidden_states_a: torch.Tensor,
        mask_a: torch.Tensor,
        hidden_states_b: torch.Tensor,
        mask_b: torch.Tensor,
        reg: float = 0.1,
        num_iters: int = 20,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Calculates the Optimal Transport (Sinkhorn) Loss between two batches of sentences 
        based on Cosine Distance. The function is highly optimized for GPU via 
        Vectorization and is absolutely safe from numerical underflow.

        Parameters:
        -----------
        hidden_states_a : torch.Tensor
            Output from the Encoder containing the semantic representations of batch A. 
            Shape: [B, T1, D] (B: Batch size, T1: Max sequence length A, D: Hidden dimension).
            Note: You do not need to normalize this tensor before passing it to the function.
            
        mask_a : torch.Tensor
            Attention mask matrix for batch A from the Tokenizer. 
            Shape: [B, T1]. 
            Values: 1 (valid token, cost will be computed), 0 (padding, will be ignored).
            
        hidden_states_b : torch.Tensor
            Output from the Encoder containing the semantic representations of batch B. 
            Shape: [B, T2, D] (T2: Max sequence length B).
            
        mask_b : torch.Tensor
            Attention mask matrix for batch B. 
            Shape: [B, T2].
            
        reg : float, default = 0.1
            Entropy regularization coefficient. Controls the "blurriness" of the Sinkhorn algorithm.
            - Too small (< 0.05): Prone to unstable/jagged gradients.
            - Too large (> 0.5): Blurs the transport matrix too much, skewing semantic alignment.
            - Recommended: 0.05 to 0.1 when using Cosine Distance.
            
        num_iters : int, default = 20
            Fixed number of iterations for the Sinkhorn algorithm to converge. 
            20 iterations is an optimal balance between computation speed and accuracy.
            
        eps : float, default = 1e-8
            A tiny value (Epsilon) added to denominators to prevent Division by Zero errors, 
            keeping the computation graph stable.

        Returns:
        --------
        torch.Tensor
            A scalar tensor representing the average Optimal Transport cost of the 
            entire Batch. You can directly call `.backward()` on it.
        """
        
        B, T1, D = hidden_states_a.shape
        T2 = hidden_states_b.shape[1]

        # 1. COMPUTE COSINE DISTANCE (COST MATRIX)
        # Normalize the vectors to length 1 (L2 Normalize)
        norm_a = F.normalize(hidden_states_a, p=2, dim=-1)
        norm_b = F.normalize(hidden_states_b, p=2, dim=-1)
        
        # Compute Cosine Similarity -> Cosine Distance. Values are bounded in [0, 2]
        cos_sim = torch.bmm(norm_a, norm_b.transpose(1, 2))
        cost = 1.0 - cos_sim

        # 2. HANDLE PADDING
        # Create a cross-attention mask between the two sentences
        valid = mask_a.unsqueeze(2) * mask_b.unsqueeze(1)
        
        # Assign a cost of 10.0 to padding positions (since the actual max cost is only 2.0)
        cost_masked = cost.masked_fill(valid == 0, 10.0)

        # 3. SINKHORN ALGORITHM
        # Compute marginal distributions (distribute mass uniformly across valid tokens)
        a = mask_a.float()
        a = a / (a.sum(dim=1, keepdim=True) + eps)

        b = mask_b.float()
        b = b / (b.sum(dim=1, keepdim=True) + eps)

        # Kernel matrix
        K = torch.exp(-cost_masked / reg)

        u = torch.ones_like(a)

        # Sinkhorn iteration loop
        for _ in range(num_iters):
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + eps)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + eps)

        # Transport Plan: [B, T1, T2]
        P = u.unsqueeze(2) * K * v.unsqueeze(1)

        # 4. COMPUTE TOTAL LOSS
        # Multiply the transport plan by the original cost and only sum over valid tokens
        ot_loss = (P * cost * valid).sum(dim=(1, 2)).mean()

        return ot_loss
    
    def freeze_model(self, freeze_llm=True, freeze_decoder=True, freeze_decoder_cross_attn=True, freeze_mt_lm_head=True):
        # freeze mt encoder
        for name, param in self.mt_model.get_encoder().named_parameters():
            param.requires_grad = False
        
        # freeze mt decoder
        if freeze_decoder:
            for name, param in self.mt_model.get_decoder().named_parameters():
                param.requires_grad = False
                if 'encoder_attn' in name and not freeze_decoder_cross_attn:      # train decoder cross-attention
                    param.requires_grad = True
                    print(f"Unfroze: {name}")

        # freeze mt lm head
        if freeze_mt_lm_head:
            for name, param in self.mt_model.lm_head.named_parameters():
                param.requires_grad = False

        # freeze llm
        if freeze_llm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

    def prepare_inputs_for_generation(
        self, 
        input_ids, # decoder's
        past_key_values=None, 
        attention_mask=None,  # encoder's 
        inputs_embeds=None, 
        cache_position=None, 
        decoder_attention_mask=None,
        encoder_outputs=None,
        position_ids=None,
        **kwargs
    ):     
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values[0], Cache):
                # past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                past_length = past_key_values[0].get_seq_length()
            else:
                past_length = past_key_values[0][0][0].shape[2]
            
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        
        if decoder_attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            ## reset position
            position_ids = decoder_attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(decoder_attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        # first inference step: src + tgt length
        # subsequent inference step: new generated tokens length, default is 1
        # print(f"position_ids shape: {position_ids}, input_ids shape: {input_ids.shape}")
        input_length = position_ids.shape[-1]
            
        ## overwrite the default updata
        cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)

        model_inputs.update(
            {   
                "decoder_input_ids": input_ids,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "encoder_outputs": encoder_outputs
            }
        )
        return model_inputs
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, 
        inputs_tensor: torch.Tensor, 
        model_kwargs, 
        model_input_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        ### default will not reture encoder's all hidden states, so we overwrite it.

        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        ## new
        encoder_kwargs["use_cache"] = False
        encoder_kwargs["output_hidden_states"] = True
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

