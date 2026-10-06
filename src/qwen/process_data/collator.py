import random
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorForLamate:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        # print("Collator for Lamate!")
        if return_tensors is None:
            return_tensors = self.return_tensors

        pad_token_id = self.tokenizer.pad_token_id
        input_ids = [feature["input_ids"] for feature in features]
        max_length = max(len(l) for l in input_ids)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        # left padding
        input_ids = [[pad_token_id]*(max_length-len(ids)) + ids for ids in input_ids]
        attention_mask = [[0 if x == pad_token_id else 1 for x in y] for y in input_ids]
        # print(features)

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        ## for predict only
        if labels is None:
            features = {
                "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
                "attention_mask": torch.tensor(np.array(attention_mask).astype(np.int64))
            }
            return features
        
        ## add eos to the end of labels
        if labels[0][-1] != self.tokenizer.eos_token_id:
            labels = [label + [self.tokenizer.eos_token_id] for label in labels]

        ## add bos to the start of labels
        if labels[0][0] != self.tokenizer.bos_token_id:
            labels = [[self.tokenizer.bos_token_id] + label for label in labels]

        ## for autogressive training
        decoder_input_ids = [label[:-1] for label in labels]
        labels = [label[1:] for label in labels]

        ## padding decoder_input_ids with right side
        max_length = max(len(l) for l in decoder_input_ids)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of


        decoder_input_ids = [ids + [pad_token_id]*(max_length-len(ids)) for ids in decoder_input_ids]
        decoder_attention_mask = [[0 if x == pad_token_id else 1 for x in y] for y in decoder_input_ids]

        ## padding labels
        labels = [label + [pad_token_id]*(max_length-len(label)) for label in labels]
        labels = [[self.label_pad_token_id if x == pad_token_id else x for x in y] for y in labels]

        # print(f"Collator: {decoder_input_ids}")

        # if not hasattr(self, "has_printed_debug"):
        #     print("\n" + "🔥" * 20)
        #     print(">>> [COLLATOR DEBUG] OUTPUT CỦA BATCH ĐẦU TIÊN:")
            
        #     # 1. Kiểm tra các Keys output
        #     # print(f"👉 Keys trả về: {list(features.keys())}")
            
        #     # 2. Kiểm tra decoder_input_ids
        #     if "decoder_input_ids" in features:
        #         dec_input = features["decoder_input_ids"]
        #         labels = features["labels"]
        #         print(f"✅ decoder_input_ids CÓ TỒN TẠI.")
        #         print(f"   Shape: {dec_input.shape}")
        #         print(f"   Dtype: {dec_input.dtype}")
        #         print(f"   Mẫu dữ liệu (dòng 0): {dec_input[0]}")
        #         print(f" {labels.shape}")
                
        #         # Check xem có bị None/NaN không (dù tensor khó bị None nhưng check giá trị lạ)
        #         if torch.isnan(dec_input).any():
        #             print("❌ CẢNH BÁO: Có giá trị NaN trong decoder_input_ids")
        #     else:
        #         print("❌ LỖI TO: Không tìm thấy key 'decoder_input_ids' trong output!")
        #         print("   (Lý do: Có thể đầu vào thiếu 'labels' nên code nhảy vào nhánh Inference)")

        #     # 3. Kiểm tra Labels (Target)
        #     if "labels" in features:
        #         print(f"✅ labels CÓ TỒN TẠI. Shape: {features['labels'].shape}")
        #     else:
        #         print("❌ warning: Không có 'labels'")

        #     print("🔥" * 20 + "\n")
            
        #     # Đánh dấu là đã in rồi, các batch sau sẽ không in nữa
        #     self.has_printed_debug = True

        features = {
            "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
            "attention_mask": torch.tensor(np.array(attention_mask).astype(np.int64)),
            "decoder_input_ids": torch.tensor(np.array(decoder_input_ids).astype(np.int64)),
            "decoder_attention_mask": torch.tensor(np.array(decoder_attention_mask).astype(np.int64)),
            "labels": torch.tensor(np.array(labels).astype(np.int64)),
        }
        # print(f"Decoder shape: {features['labels'].shape}")
        # print(f"Labels shape: {features['decoder_input_ids'].shape}")

        return features
    
@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        print("DataCollatorForCausalLM")
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature['labels'] for feature in features] if "labels" in features[0].keys() else None
        input_ids = [feature["input_ids"] for feature in features]

        max_length = max(len(l) for l in input_ids)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of


        input_ids =  [[self.tokenizer.pad_token_id]*(max_length-len(ids)) + ids for ids in input_ids]
        labels = [[-100]*(max_length-len(ids))+ids for ids in labels] if labels is not None else None
        attention_mask = [[0 if x == self.tokenizer.pad_token_id else 1 for x in y] for y in input_ids]

        features = {
            "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
            "attention_mask": torch.tensor(np.array(attention_mask).astype(np.int64))
        }

        if labels is not None:
            features["labels"] = torch.tensor(np.array(labels).astype(np.int64)) 

        return features
    
@dataclass
class DataCollatorForQwenNLLB:
    llm_tokenizer: PreTrainedTokenizerBase
    seq2seq_tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        # print("Collator for Lamate!")
        if return_tensors is None:
            return_tensors = self.return_tensors

        encoder_pad_token_id = self.llm_tokenizer.pad_token_id
        decoder_pad_token_id = self.seq2seq_tokenizer.pad_token_id
        input_ids = [feature["input_ids"] for feature in features]
        max_length = max(len(l) for l in input_ids)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        # left padding
        input_ids = [[encoder_pad_token_id]*(max_length-len(ids)) + ids for ids in input_ids]
        attention_mask = [[0 if x == encoder_pad_token_id else 1 for x in y] for y in input_ids]
        # print(features)



        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            input_ids_2 = [feature["input_ids_2"] for feature in features]
            max_length_2 = max(len(l) for l in input_ids_2)
            if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
                max_length_2 = (max_length_2 + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
            input_ids_2 = [[encoder_pad_token_id]*(max_length_2-len(ids)) + ids for ids in input_ids_2]
            attention_mask_2 = [[0 if x == encoder_pad_token_id else 1 for x in y] for y in input_ids_2]

        ## for predict only
        if labels is None:
            # print(features[0].keys())
            decoder_input_ids = [f["decoder_input_ids"] for f in features]
            features = {
                "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
                "attention_mask": torch.tensor(np.array(attention_mask).astype(np.int64)),
                "decoder_input_ids": torch.tensor(np.array(decoder_input_ids).astype(np.int64)),
            }
            return features
        
        ## add eos to the end of labels
        if labels[0][-1] != self.seq2seq_tokenizer.eos_token_id:
            labels = [label + [self.seq2seq_tokenizer.eos_token_id] for label in labels]

        ## add bos to the start of labels
        # if labels[0][0] != self.tokenizer.bos_token_id:
        #     labels = [[self.tokenizer.bos_token_id] + label for label in labels]
        # for i, label in enumerate(labels):
        #     if label[0] != features[i]["force_bos_token_id"]:
        #         label = [self.seq2seq_tokenizer.bos_token_id] + label

        ## for autogressive training
        decoder_input_ids = [label[:-1] for label in labels]
        labels = [label[1:] for label in labels]

        ## padding decoder_input_ids with right side
        max_length = max(len(l) for l in decoder_input_ids)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            max_length = (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of


        decoder_input_ids = [ids + [decoder_pad_token_id]*(max_length-len(ids)) for ids in decoder_input_ids]
        decoder_attention_mask = [[0 if x == decoder_pad_token_id else 1 for x in y] for y in decoder_input_ids]

        ## padding labels
        labels = [label + [self.label_pad_token_id]*(max_length-len(label)) for label in labels]
        labels = [[self.label_pad_token_id if x == self.label_pad_token_id else x for x in y] for y in labels]

        # print(f"Collator: {decoder_input_ids}")

        # if not hasattr(self, "has_printed_debug"):
        #     print("\n" + "🔥" * 20)
        #     print(">>> [COLLATOR DEBUG] OUTPUT CỦA BATCH ĐẦU TIÊN:")
            
        #     # 1. Kiểm tra các Keys output
        #     # print(f"👉 Keys trả về: {list(features.keys())}")
            
        #     # 2. Kiểm tra decoder_input_ids
        #     if "decoder_input_ids" in features:
        #         dec_input = features["decoder_input_ids"]
        #         labels = features["labels"]
        #         print(f"✅ decoder_input_ids CÓ TỒN TẠI.")
        #         print(f"   Shape: {dec_input.shape}")
        #         print(f"   Dtype: {dec_input.dtype}")
        #         print(f"   Mẫu dữ liệu (dòng 0): {dec_input[0]}")
        #         print(f" {labels.shape}")
                
        #         # Check xem có bị None/NaN không (dù tensor khó bị None nhưng check giá trị lạ)
        #         if torch.isnan(dec_input).any():
        #             print("❌ CẢNH BÁO: Có giá trị NaN trong decoder_input_ids")
        #     else:
        #         print("❌ LỖI TO: Không tìm thấy key 'decoder_input_ids' trong output!")
        #         print("   (Lý do: Có thể đầu vào thiếu 'labels' nên code nhảy vào nhánh Inference)")

        #     # 3. Kiểm tra Labels (Target)
        #     if "labels" in features:
        #         print(f"✅ labels CÓ TỒN TẠI. Shape: {features['labels'].shape}")
        #     else:
        #         print("❌ warning: Không có 'labels'")

        #     print("🔥" * 20 + "\n")
            
        #     # Đánh dấu là đã in rồi, các batch sau sẽ không in nữa
        #     self.has_printed_debug = True

        features = {
            "input_ids": torch.tensor(np.array(input_ids).astype(np.int64)),
            "attention_mask": torch.tensor(np.array(attention_mask).astype(bool)),
            "decoder_input_ids": torch.tensor(np.array(decoder_input_ids).astype(np.int64)),
            "decoder_attention_mask": torch.tensor(np.array(decoder_attention_mask).astype(bool)),
            "labels": torch.tensor(np.array(labels).astype(np.int64)),
            "input_ids_2": torch.tensor(np.array(input_ids_2).astype(np.int64)),
            "attention_mask_2": torch.tensor(np.array(attention_mask_2).astype(bool)),
        }
        # print(f"Decoder shape: {features['labels'].shape}")
        # print(f"Labels shape: {features['decoder_input_ids'].shape}")

        return features

