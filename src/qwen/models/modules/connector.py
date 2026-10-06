import torch
from torch import nn
import copy
from typing import List, Optional, Tuple, Union, Dict, Any

from transformers.modeling_utils import PreTrainedModel
from transformers import  PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from .normalization import QwenRMSNorm
from qwen.models.encoder import QwenModelBiAttEncoder

class ProjectDown(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.linear_1 = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(decoder_dim, decoder_dim, bias=False)

        # --- KHỞI TẠO TRỌNG SỐ ---
        self.apply(self._init_weights)


    def _init_weights(self, module):
            """Khởi tạo Kaiming Normal (tốt cho GELU/ReLU)"""
            if isinstance(module, nn.Linear):
                # fan_in: Bảo toàn độ lớn phương sai trong quá trình Forward pass
                # nonlinearity='relu': Phù hợp với GELU (vì GELU ~ ReLU ở miền dương)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            # Mẹo: Lớp output cuối cùng (linear_2) nên được khởi tạo nhỏ hơn một chút
            # để mạng bắt đầu học từ những bước nhỏ, tránh shock loss đầu tiên.
            if module == self.linear_2:
                # Scale lại weight nhỏ đi (ví dụ nhân 0.02)
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        
        return x
    
## MLP + TinyEncoder
class Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder_config = config.decoder

        if isinstance(decoder_config, dict):
            config.decoder = PretrainedConfig.from_dict(
                decoder_config
            )

        self.encoder_project = ProjectDown(encoder_dim=config.hidden_size, decoder_dim=config.decoder.hidden_size)
        # self.encoder_project = ProjectDown(encoder_dim=4096, decoder_dim=2048)
        self.post_encoder = None
        self.encoder_method = getattr(config, "encoder_method", "causal")

        if self.encoder_method == "stack":
            tiny_encoder_config = copy.deepcopy(config.decoder)
            # tiny_encoder_config["num_hidden_layers"] = getattr(config, "encoder_layer_num", 8)
            tiny_encoder_config.num_hidden_layers = getattr(config, "encoder_layer_num", 8)
            tiny_encoder_config.layer_types = ["full_attention"] * tiny_encoder_config.num_hidden_layers
            # stack_encoder_config = PretrainedConfig.from_dict(tiny_encoder_config)  
            # stack_encoder_config = PretrainedConfig.from_dict(tiny_encoder_config.to_dict())  
            self.post_encoder = QwenModelBiAttEncoder(tiny_encoder_config)
            self.post_encoder.apply(self.post_encoder._init_weights)
      
    def forward(
        self, 
        hidden,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden = self.encoder_project(hidden)
        hidden_states = (hidden,)        
        if self.encoder_method == "stack":
            post_encoder_outputs = self.post_encoder(
                attention_mask=attention_mask,
                inputs_embeds=hidden,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            hidden = post_encoder_outputs.last_hidden_state
            hidden_states += post_encoder_outputs.hidden_states
        # print(len(hidden_states))
            
        ## temp return None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )


class GroupedEncoderFusion(nn.Module):
    def __init__(self, config, group_size):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.group_size = group_size
        assert self.num_layers % group_size == 0, "Total number of layers must be divisible by the group size"
        
        self.num_groups = self.num_layers // self.group_size
        self.weights = nn.Parameter(torch.randn(self.num_groups, 1, 1, 1))
        # Layer Normalization
        self.layer_norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states):
        """
        hidden_states: List of tensors, each tensor is of shape (batch_size, seq_length, hidden_size)
        """
        # hidden_stack = torch.stack(hidden_states, dim=0)  # Shape: [num_layers, batch_size, seq_length, hidden_size]
        # last_layers = hidden_stack[self.group_size-1::self.group_size]  # Shape: [num_groups, batch_size, seq_length, hidden_size]

        last_layers = []
        for i in range(self.group_size - 1, len(hidden_states), self.group_size):
            last_layers.append(hidden_states[i])
        last_layers = torch.stack(last_layers, dim=0)
       
        normalized_weights = torch.sigmoid(self.weights)  # Shape: [num_groups, 1, 1]
        weighted_sum = torch.sum(last_layers * normalized_weights, dim=0) / self.num_groups # Shape: [batch_size, seq_length, hidden_size]
        
        fused_hidden = self.layer_norm(weighted_sum)
        
        return fused_hidden