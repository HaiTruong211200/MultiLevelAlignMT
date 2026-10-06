import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import inspect
from dataclasses import dataclass

from transformers import  PretrainedConfig, AutoModelForCausalLM
from transformers import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput,  ModelOutput
from transformers.utils import is_accelerate_available
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .base_model import QwenPreTrainedModel
from .encoder import QwenModelEncoder
from .combine_encoder import QwenModelCombineEncoder
from .decoder import QwenCrossAttDecoder
from qwen.utils.utils import print_train_module


@dataclass
class CustomSeq2SeqLMOutput(Seq2SeqLMOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    ot_loss: Optional[torch.FloatTensor] = None


if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module


class QwenForEncDec(QwenPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        encoder_method = getattr(config, "encoder_method", "causal")
        self.contrastive_lambda = getattr(config, "contrastive_lambda", 0.0)
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.1)
        self.ot_lambda = getattr(config, "ot_lambda", 0.0)
        self.ot_reg = getattr(config, "ot_reg", 0.1)
        self.ot_num_iters = getattr(config, "ot_num_iters", 20)
        self.ot_eps = getattr(config, "ot_eps", 1e-8)
        self.num_layers_align = getattr(config, "num_layers_align", 4)
        print(f"Contrastive Lambda: {self.contrastive_lambda}, Contrastive Temperature: {self.contrastive_temperature}")
        print(f"OT Lambda: {self.ot_lambda}, OT Reg: {self.ot_reg}, OT Num Iters: {self.ot_num_iters}, OT Eps: {self.ot_eps}")

        ## encoder
        if encoder_method == "causal":
            self.encoder = QwenModelEncoder(config)
            print("Using causal LLM Encoder!")
        # elif encoder_method == "bidirectional":
        #     self.encoder = QwenModelBiAttEncoder(config)
        #     print("Using bidirectional LLM Encoder!")
        elif encoder_method in ["stack", "project"]: 
            print("Using combined LLM Encoder!")
            self.encoder = QwenModelCombineEncoder(config)
        self.lm_head = None

        ## decoder
        # decoder_config = PretrainedConfig.from_dict(config.decoder) if getattr(config, "decoder", None) is not None else config
        decoder_config = PretrainedConfig.from_dict(config.decoder.to_dict()) if getattr(config, "decoder", None) is not None else config
        if decoder_config.model_method in ["lamate", "SailorED"]:
            self.decoder = QwenCrossAttDecoder(decoder_config)
            self.lm_head = nn.Linear(decoder_config.hidden_size, config.vocab_size, bias=False)
        else:
            print("Not implement this model yet!")
            exit()

        if self.lm_head is None:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.llm.model.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.llm.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
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

    ## referenced from T5
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # need encoder return all hidden state
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

        # a tuple of length layer_num + 1 
        encoder_all_hidden_states = encoder_outputs.hidden_states
        # encoder_last_hidden_state = encoder_outputs.last_hidden_state
        if labels is not None:
            dec_hidden_states = self.encoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states

        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_all_hidden_states=encoder_all_hidden_states,
            encoder_attention_mask=attention_mask,
            # use_cache=use_cache,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        hidden_states = decoder_outputs[0]
        ## compute loss 
        pretraining_tp = getattr(self.config, "pretraining_tp", 1)
        if pretraining_tp is not None and pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = lm_loss
            if self.contrastive_lambda > 0:
                contrastive_loss = 0
                for i in range(1, self.num_layers_align + 1):
                    contrastive_loss += self.compute_contrastive_loss(encoder_all_hidden_states[-i], dec_hidden_states[-i])
                loss = lm_loss + self.contrastive_lambda * contrastive_loss / self.num_layers_align
            if self.ot_lambda > 0:
                ot_loss = self.compute_ot_loss_cosine(
                    hidden_states_a=encoder_all_hidden_states[-1],
                    mask_a=attention_mask,
                    hidden_states_b=dec_hidden_states[-1],
                    mask_b=decoder_attention_mask,
                    reg=self.ot_reg,
                    num_iters=self.ot_num_iters,
                    eps=self.ot_eps
                )

                loss = loss + self.ot_lambda * ot_loss
        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return (loss,) + output if loss is not None else output

        if labels is None:
            return CustomSeq2SeqLMOutput(
                loss=loss,
                lm_loss=None,
                contrastive_loss=None,
                ot_loss=None,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            return CustomSeq2SeqLMOutput(
                loss=loss,
                lm_loss=lm_loss.detach(),
                contrastive_loss=contrastive_loss.detach() if self.contrastive_lambda > 0 else None,
                ot_loss=ot_loss.detach() if self.ot_lambda > 0 else None,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

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

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: Union[int, List[int]] = None,
        bos_token_id: int = None,
        device: torch.device = None,
        *args,
        **kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        ## add decoder attention mask to model_kwargs

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        # -----------------------------------------------------------
        # 2. TÌM TOKEN BẮT ĐẦU (🔴 ĐÃ SỬA LẠI ĐOẠN NÀY)
        # -----------------------------------------------------------
        # Thay vì gọi hàm self._get_decoder_start_token_id(...) bị thiếu
        # Chúng ta tự viết logic tìm ID ngay tại đây:
        
        # Ưu tiên 1: Lấy từ generation_config
        if decoder_start_token_id is None and hasattr(self, "generation_config") and self.generation_config is not None:
            decoder_start_token_id = self.generation_config.decoder_start_token_id
            
        # Ưu tiên 2: Lấy từ config model gốc
        if decoder_start_token_id is None and hasattr(self, "config"):
            if hasattr(self.config, "decoder_start_token_id"):
                 decoder_start_token_id = self.config.decoder_start_token_id
            elif hasattr(self.config, "bos_token_id"):
                 decoder_start_token_id = self.config.bos_token_id
        
        # Ưu tiên 3: Lấy từ tham số bos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        # Kiểm tra sống còn
        if decoder_start_token_id is None:
             raise ValueError(
                "Không tìm thấy `decoder_start_token_id`. "
                "Hãy khai báo nó trong `generation_config` hoặc `config.json`!"
            )
        if device is None:
            device = self.device
        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expcted to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long, device=device)
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            decoder_input_ids_start = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
            model_kwargs["decoder_attention_mask"] = torch.ones_like(decoder_input_ids)
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]
        Default will not expand encoder's all hidden states, so we overwrite it.
        """
        ## default will not expand hidden_states with tuple type, so we overwrite it.

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
                ## a tuple
                elif key == "hidden_states" and dict_to_expand[key] is not None:
                    dict_to_expand[key] = tuple(x.repeat_interleave(expand_size, dim=0) for x in dict_to_expand[key])
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
class QwenCrossAttentionEncDec(QwenForEncDec, GenerationMixin):
    def __init__(self, config):
        if hasattr(config, "decoder") and isinstance(config.decoder, dict):
            target_layers = config.decoder.get("num_hidden_layers", None)
            
            # 2. Nếu có danh sách layer_types bị thừa, hãy cắt bớt nó đi
            if target_layers is not None and "layer_types" in config.decoder:
                current_types = config.decoder["layer_types"]
                if len(current_types) > target_layers:
                    print(f"Fixing layer_types: Truncating from {len(current_types)} to {target_layers}")
                    config.decoder["layer_types"] = current_types[:target_layers]
            config.decoder = Qwen2Config(**config.decoder)
            
        # Kiểm tra tương tự cho encoder (nếu cần)
        if hasattr(config, "encoder") and isinstance(config.encoder, dict):
            config.encoder = Qwen2Config(**config.encoder)
        super().__init__(config)

    def freeze_llm(self):
        for name, param in self.named_parameters():
            is_freeze = False

            ## freeze the whole encoder except connector
            if name.startswith("encoder.") and not name.startswith("encoder.connector") and not name.startswith("encoder.fuse_model"):
                param.requires_grad = False
                is_freeze = True
            if torch.cuda.current_device() == 0 and is_freeze:
                print(f"freeze ==> {name}")
        # print_train_module(self)

    def freeze_decoder(self, freeze_cross_attn=True):
        for name, param in self.named_parameters():
            is_freeze = False

            if name.startswith("decoder."):
                # freeze toàn bộ decoder
                if freeze_cross_attn:
                    param.requires_grad = False
                    is_freeze = True

                # freeze decoder nhưng giữ cross attention trainable
                else:
                    if ".cross_attn." not in name:
                        param.requires_grad = False
                        is_freeze = True

            if torch.cuda.current_device() == 0 and is_freeze:
                print(f"freeze ==> {name}")

    
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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for curent_kv in past_key_values:
            reordered_curent = ()
            for layer_past in curent_kv:
                reordered_curent += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            reordered_past += (reordered_curent,)
        return reordered_past
class QwenEncDecNLLB(QwenPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        encoder_method = getattr(
            config,
            "encoder_method",
            "causal"
        )

        # OT / contrastive params
        self.contrastive_lambda = getattr(
            config,
            "contrastive_lambda",
            0.0
        )

        self.contrastive_temperature = getattr(
            config,
            "contrastive_temperature",
            0.1
        )

        self.ot_lambda = getattr(config, "ot_lambda", 0.0)
        self.ot_reg = getattr(config, "ot_reg", 0.1)
        self.ot_num_iters = getattr(config, "ot_num_iters", 20)
        self.ot_eps = getattr(config, "ot_eps", 1e-8)

        # ====================
        # Qwen encoder
        # ====================

        if encoder_method == "causal":
            print("Using causal LLM Encoder!")
            self.encoder = QwenModelEncoder(config)

        elif encoder_method in ["stack", "project"]:
            print("Using combined LLM Encoder!")
            self.encoder = QwenModelCombineEncoder(config)

        else:
            raise ValueError(
                f"Unsupported encoder_method: {encoder_method}"
            )

        # ====================
        # NLLB decoder
        # ====================

        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(config.decoder.model_name_or_path)

        self.vocab_size = self.mt_model.config.vocab_size

        # tied embeddings
        self._dynamic_tied_weights_keys = [
            "decoder.embed_tokens.weight"
        ]

    def get_input_embeddings(self):
        return self.mt_model.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.mt_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.mt_model.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.mt_model.model.decoder

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
            


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # need encoder return all hidden state

        if labels is not None:
            # if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            # a tuple of length layer_num + 1 
            encoder_all_hidden_states = encoder_outputs.hidden_states
            # encoder_last_hidden_state = encoder_outputs.last_hidden_state

            # Contrastive Loss
            # dec_attention = torch.ones_like(decoder_input_ids)
            # dec_hidden_states = self.encoder(
            #     input_ids=decoder_input_ids,
            #     attention_mask=dec_attention,
            #     inputs_embeds=decoder_inputs_embeds,
            #     use_cache=False,
            #     output_attentions=False,
            #     output_hidden_states=True,
            #     return_dict=True,
            # ).hidden_states[-1]

            position_ids = position_ids if position_ids is not None else torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device).unsqueeze(0).expand(decoder_input_ids.shape[0], -1)


            # decoder_outputs = self.decoder(
            #     decoder_input_ids,
            #     attention_mask=decoder_attention_mask,
            #     inputs_embeds=decoder_inputs_embeds,
            #     past_key_values=past_key_values,
            #     encoder_all_hidden_states=encoder_all_hidden_states,
            #     encoder_attention_mask=attention_mask,
            #     # use_cache=use_cache,
            #     use_cache=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            #     cache_position=cache_position,
            #     position_ids=position_ids,
            # )

            decoder_outputs = self.mt_model.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_all_hidden_states[-1],
                encoder_attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                # position_ids=position_ids,
            )
            # print(decoder_outputs.keys())
            hidden_states = decoder_outputs[0]
            ## compute loss 
            pretraining_tp = getattr(self.config, "pretraining_tp", 1)
            if pretraining_tp is not None and pretraining_tp > 1:
                print(f"Using pretraining_tp={pretraining_tp} for parallel LM head computation!")
                lm_head_slices = self.mt_model.lm_head.weight.split(self.vocab_size // pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.mt_model.lm_head(hidden_states)
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
                    contrastive_loss = self.compute_contrastive_loss(encoder_all_hidden_states[-1], hidden_states)
                    # contrastive_loss = self.compute_contrastive_loss(encoder_all_hidden_states[-1], dec_hidden_states)
                    loss = lm_loss + self.contrastive_lambda * contrastive_loss
                if self.ot_lambda > 0:
                    ot_loss = self.compute_ot_loss_cosine(
                        encoder_hidden_states=encoder_all_hidden_states[-1],
                        mask_a=attention_mask,
                        hidden_states_b=hidden_states,
                        mask_b=decoder_attention_mask,
                        reg=self.ot_reg,
                        num_iters=self.ot_num_iters,
                        eps=self.ot_eps
                    )
                    print(f"OT Loss: {ot_loss.item():.4f}")
                    loss = loss + self.ot_lambda * ot_loss
                # else:
                #     loss = lm_loss
            
            # import pdb;pdb.set_trace()
            if not return_dict:
                output = (logits,) + decoder_outputs[1:]
                return (loss,) + output if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                # decoder_hidden_states=decoder_outputs,
                cross_attentions=None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
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

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: Union[int, List[int]] = None,
        bos_token_id: int = None,
        device: torch.device = None,
        *args,
        **kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        ## add decoder attention mask to model_kwargs

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        # print(f"model_kwargs keys: {model_kwargs.keys()}")
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # print("Getting decoder_input_ids from model_kwargs['decoder_input_ids']")
            # print(f"decoder_input_ids shape: {model_kwargs['decoder_input_ids'].shape}")
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if model_kwargs.get("decoder_attention_mask") is None:
                model_kwargs["decoder_attention_mask"] = torch.ones_like(decoder_input_ids)
            return decoder_input_ids, model_kwargs
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
            if model_kwargs.get("decoder_attention_mask") is None:
                model_kwargs["decoder_attention_mask"] = torch.ones_like(decoder_input_ids)
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        # -----------------------------------------------------------
        # 2. TÌM TOKEN BẮT ĐẦU (🔴 ĐÃ SỬA LẠI ĐOẠN NÀY)
        # -----------------------------------------------------------
        # Thay vì gọi hàm self._get_decoder_start_token_id(...) bị thiếu
        # Chúng ta tự viết logic tìm ID ngay tại đây:
        
        # Ưu tiên 1: Lấy từ generation_config
        # if decoder_start_token_id is None and hasattr(self, "generation_config") and self.generation_config is not None:
        #     decoder_start_token_id = self.generation_config.decoder_start_token_id
            
        # # Ưu tiên 2: Lấy từ config model gốc
        # if decoder_start_token_id is None and hasattr(self, "config"):
        #     if hasattr(self.config, "decoder_start_token_id"):
        #          decoder_start_token_id = self.config.decoder_start_token_id
        #     elif hasattr(self.config, "bos_token_id"):
        #          decoder_start_token_id = self.config.bos_token_id
        
        # # Ưu tiên 3: Lấy từ tham số bos_token_id
        # if decoder_start_token_id is None:
        #     decoder_start_token_id = bos_token_id

        # # Kiểm tra sống còn
        # if decoder_start_token_id is None:
        #      raise ValueError(
        #         "Không tìm thấy `decoder_start_token_id`. "
        #         "Hãy khai báo nó trong `generation_config` hoặc `config.json`!"
        #     )
        if device is None:
            device = self.device
        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expcted to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long, device=device)
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            decoder_input_ids_start = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
            model_kwargs["decoder_attention_mask"] = torch.ones_like(decoder_input_ids)
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]
        Default will not expand encoder's all hidden states, so we overwrite it.
        """
        ## default will not expand hidden_states with tuple type, so we overwrite it.

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
                ## a tuple
                elif key == "hidden_states" and dict_to_expand[key] is not None:
                    dict_to_expand[key] = tuple(x.repeat_interleave(expand_size, dim=0) for x in dict_to_expand[key])
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    

class QwenCrossAttentionEncDecNLLB(QwenEncDecNLLB, GenerationMixin):
    def __init__(self, config):
        # if hasattr(config, "decoder") and isinstance(config.decoder, dict):
        #     target_layers = config.decoder.get("num_hidden_layers", None)
            
        #     # 2. Nếu có danh sách layer_types bị thừa, hãy cắt bớt nó đi
        #     if target_layers is not None and "layer_types" in config.decoder:
        #         current_types = config.decoder["layer_types"]
        #         if len(current_types) > target_layers:
        #             print(f"Fixing layer_types: Truncating from {len(current_types)} to {target_layers}")
        #             config.decoder["layer_types"] = current_types[:target_layers]
        #     config.decoder = Qwen2Config(**config.decoder)
            
        # Kiểm tra tương tự cho encoder (nếu cần)
        if hasattr(config, "encoder") and isinstance(config.encoder, dict):
            config.encoder = Qwen2Config(**config.encoder)
        super().__init__(config)

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
            for name, param in self.encoder.named_parameters():
                is_freeze = False
                if "connector" not in name and "fuse_model" not in name and "embed_tokens" not in name:
                    param.requires_grad = False
                    is_freeze = True
                # if torch.cuda.current_device() == 0 and is_freeze:
                #     print(f"freeze ==> {name}")

        

        # for name, param in self.named_parameters():
        #     is_freeze = False

        #     ## freeze the whole encoder except connector
        #     if freeze_llm:
        #         if name.startswith("encoder.") and not name.startswith("encoder.connector") and not name.startswith("encoder.fuse_model"):
        #             param.requires_grad = False
        #             is_freeze = True
        #     if name.startswith("mt_model.model.shared") and freeze_decoder:
        #         param.requires_grad = False
        #         is_freeze = True

        #     if name.startswith("mt_model.lm_head"):
        #         param.requires_grad = True
        #         is_freeze = False

            

        #     if name.startswith("mt_model.model.encoder"):
        #         param.requires_grad = False
        #         is_freeze = True
        #         # freeze toàn bộ decoder
        #     elif name.startswith("mt_model.model.decoder") and freeze_decoder:
        #         if not freeze_decoder_cross_attn:
        #             if "encoder_attn" not in name:
        #                 param.requires_grad = False
        #                 is_freeze = True
        #         else:
        #             param.requires_grad = False
        #             is_freeze = True

        #     if torch.cuda.current_device() == 0 and is_freeze:
        #         print(f"freeze ==> {name}")
        # print_train_module(self)

    # def freeze_encoder(self):
    #     for name, param in self.named_parameters():
    #         is_freeze = False

    #         ## freeze the whole encoder except connector
    #         if name.startswith("decoder.model.encoder"):
    #             param.requires_grad = False
    #             is_freeze = True

    #         if torch.cuda.current_device() == 0 and is_freeze:
    #             print(f"freeze ==> {name}")
    #     print_train_module(self)
    
    def freeze_mt(self, freeze_decoder=True, freeze_cross_attn=True):
        for name, param in self.named_parameters():
            is_freeze = False

            if name.startswith("mt_model.model.encoder"):
                param.requires_grad = False
                is_freeze = True
                # freeze toàn bộ decoder
            elif name.startswith("mt_model.model.decoder") and freeze_decoder:
                if freeze_cross_attn:
                    if "encoder_attn" not in name:
                        param.requires_grad = False
                        is_freeze = True
                else:
                    param.requires_grad = False
                    is_freeze = True
                #     if freeze_cross_attn:
                #         param.requires_grad = False
                #         is_freeze = True

                # # freeze decoder nhưng giữ cross attention trainable
                #     else:
                #         if ".cross_attn." not in name:
                #             param.requires_grad = False
                #             is_freeze = True

            if torch.cuda.current_device() == 0 and is_freeze:
                print(f"freeze ==> {name}")


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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for curent_kv in past_key_values:
            reordered_curent = ()
            for layer_past in curent_kv:
                reordered_curent += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            reordered_past += (reordered_curent,)
        return reordered_past

class QwenforEncDec(QwenPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        encoder_method = getattr(config, "encoder_method", "causal")
        self.contrastive_lambda = getattr(config, "contrastive_lambda", 0.0)
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.1)

        ## encoder
        if encoder_method == "causal":
            self.encoder = QwenModelEncoder(config)
            print("Using causal LLM Encoder!")
        # elif encoder_method == "bidirectional":
        #     self.encoder = QwenModelBiAttEncoder(config)
        #     print("Using bidirectional LLM Encoder!")
        elif encoder_method in ["stack", "project"]: 
            print("Using combined LLM Encoder!")
            self.encoder = QwenModelCombineEncoder(config)
        self.lm_head = None

        ## decoder
        # decoder_config = PretrainedConfig.from_dict(config.decoder) if getattr(config, "decoder", None) is not None else config
        decoder_config = PretrainedConfig.from_dict(config.decoder.to_dict()) if getattr(config, "decoder", None) is not None else config
        if decoder_config.model_method == "SailorED":
            self.decoder = QwenCrossAttDecoder(decoder_config)
            self.lm_head = nn.Linear(decoder_config.hidden_size, config.vocab_size, bias=False)
        else:
            print("Not implement this model yet!")
            exit()

        if self.lm_head is None:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.llm = AutoModelForCausalLM.from_pretrained("sail/Sailor2-1B-Chat")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
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

    ## referenced from T5
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # need encoder return all hidden state
        if labels is not None:
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )

            # a tuple of length layer_num + 1 
            encoder_all_hidden_states = encoder_outputs.hidden_states
            # encoder_last_hidden_state = encoder_outputs.last_hidden_state

            # Contrastive Loss
            # dec_attention = torch.ones_like(decoder_input_ids)
            # dec_hidden_states = self.encoder(
            #     input_ids=decoder_input_ids,
            #     attention_mask=dec_attention,
            #     inputs_embeds=decoder_inputs_embeds,
            #     use_cache=False,
            #     output_attentions=False,
            #     output_hidden_states=True,
            #     return_dict=True,
            # ).hidden_states[-1]


            decoder_outputs = self.decoder(
                decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_all_hidden_states=encoder_all_hidden_states,
                encoder_attention_mask=attention_mask,
                # use_cache=use_cache,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                position_ids=position_ids,
            )
            hidden_states = decoder_outputs[0]
            ## compute loss 
            pretraining_tp = getattr(self.config, "pretraining_tp", 1)
            if pretraining_tp is not None and pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()
            
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                labels = labels.to(logits.device)
                lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                if self.contrastive_lambda > 0:
                    contrastive_loss = self.compute_contrastive_loss(encoder_all_hidden_states[-1], hidden_states)
                    # contrastive_loss = self.compute_contrastive_loss(encoder_all_hidden_states[-1], dec_hidden_states)
                    loss = lm_loss + self.contrastive_lambda * contrastive_loss
                else:
                    loss = lm_loss
            
            # import pdb;pdb.set_trace()
            if not return_dict:
                output = (logits,) + decoder_outputs[1:]
                return (loss,) + output if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,

                cross_attentions=None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        
        else:
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                num_beams=5,
                do_sample=False,
            )
            generated_tokens = outputs[:, input_ids.shape[1]:]
            return generated_tokens

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

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: Union[int, List[int]] = None,
        bos_token_id: int = None,
        device: torch.device = None,
        *args,
        **kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        ## add decoder attention mask to model_kwargs

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        print(f"model_kwargs keys: {model_kwargs.keys()}")
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs.keys():
            print("Getting decoder_input_ids from model_kwargs['decoder_input_ids']")
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            return decoder_input_ids, model_kwargs
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None
        


        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        # -----------------------------------------------------------
        # 2. TÌM TOKEN BẮT ĐẦU (🔴 ĐÃ SỬA LẠI ĐOẠN NÀY)
        # -----------------------------------------------------------
        # Thay vì gọi hàm self._get_decoder_start_token_id(...) bị thiếu
        # Chúng ta tự viết logic tìm ID ngay tại đây:
        
        # Ưu tiên 1: Lấy từ generation_config
        if decoder_start_token_id is None and hasattr(self, "generation_config") and self.generation_config is not None:
            decoder_start_token_id = self.generation_config.decoder_start_token_id
            
        # Ưu tiên 2: Lấy từ config model gốc
        if decoder_start_token_id is None and hasattr(self, "config"):
            if hasattr(self.config, "decoder_start_token_id"):
                 decoder_start_token_id = self.config.decoder_start_token_id
            elif hasattr(self.config, "bos_token_id"):
                 decoder_start_token_id = self.config.bos_token_id
        
        # Ưu tiên 3: Lấy từ tham số bos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        # Kiểm tra sống còn
        if decoder_start_token_id is None:
             raise ValueError(
                "Không tìm thấy `decoder_start_token_id`. "
                "Hãy khai báo nó trong `generation_config` hoặc `config.json`!"
            )
        if device is None:
            device = self.device
        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expcted to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long, device=device)
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            decoder_input_ids_start = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
            model_kwargs["decoder_attention_mask"] = torch.ones_like(decoder_input_ids)
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]
        Default will not expand encoder's all hidden states, so we overwrite it.
        """
        ## default will not expand hidden_states with tuple type, so we overwrite it.

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
                ## a tuple
                elif key == "hidden_states" and dict_to_expand[key] is not None:
                    dict_to_expand[key] = tuple(x.repeat_interleave(expand_size, dim=0) for x in dict_to_expand[key])
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class QwenCrossAtentionEncDec(QwenforEncDec, GenerationMixin):
    def __init__(self, config):
        if hasattr(config, "decoder") and isinstance(config.decoder, dict):
            target_layers = config.decoder.get("num_hidden_layers", None)
            
            # 2. Nếu có danh sách layer_types bị thừa, hãy cắt bớt nó đi
            if target_layers is not None and "layer_types" in config.decoder:
                current_types = config.decoder["layer_types"]
                if len(current_types) > target_layers:
                    print(f"Fixing layer_types: Truncating from {len(current_types)} to {target_layers}")
                    config.decoder["layer_types"] = current_types[:target_layers]
            config.decoder = Qwen2Config(**config.decoder)
            
        # Kiểm tra tương tự cho encoder (nếu cần)
        if hasattr(config, "encoder") and isinstance(config.encoder, dict):
            config.encoder = Qwen2Config(**config.encoder)
        super().__init__(config)

    def freeze_llm(self):
        for name, param in self.named_parameters():
            is_freeze = False

            ## freeze the whole encoder except connector
            if name.startswith("encoder.") and not name.startswith("encoder.connector") and not name.startswith("encoder.fuse_model"):
                param.requires_grad = False
                is_freeze = True
            if torch.cuda.current_device() == 0 and is_freeze:
                print(f"freeze ==> {name}")
        print_train_module(self)

    def freeze_decoder(self, freeze_cross_attn=True):
        for name, param in self.named_parameters():
            is_freeze = False

            if name.startswith("decoder."):
                # freeze toàn bộ decoder
                if freeze_cross_attn:
                    param.requires_grad = False
                    is_freeze = True

                # freeze decoder nhưng giữ cross attention trainable
                else:
                    if ".cross_attn." not in name:
                        param.requires_grad = False
                        is_freeze = True

            if torch.cuda.current_device() == 0 and is_freeze:
                print(f"freeze ==> {name}")

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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for curent_kv in past_key_values:
            reordered_curent = ()
            for layer_past in curent_kv:
                reordered_curent += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            reordered_past += (reordered_curent,)
        return reordered_past   