# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import os

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from verl.utils.device import get_torch_device

from .base import BaseRollout

__all__ = ["HFRollout"]

_ECHO_DEBUG = os.environ.get("ECHO_DEBUG_SEQUENCE_LENGTH", "0") == "1"


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": self.config.n,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        if _ECHO_DEBUG:
            print("=== ECHO_DEBUG: _generate_minibatch input ===")
            print(f"  prompts.batch keys: {list(prompts.batch.keys())}")
            print(f"  prompts.non_tensor_batch keys: {list(prompts.non_tensor_batch.keys())}")
            print(f"  prompts.meta_info: {prompts.meta_info}")
            print(f"  input_ids shape: {idx.shape}, dtype: {idx.dtype}")
            print(f"  attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
            print(f"  position_ids shape: {position_ids.shape}, dtype: {position_ids.dtype}")
            print(f"  prompt_length (idx.size(1)): {prompt_length}")
            print(f"  response_length (config): {response_length}")
            print(f"  expected sequence_length: {prompt_length + response_length}")
            # per-sample prompt token length (count non-pad tokens)
            for i in range(idx.size(0)):
                num_pad = (idx[i] == pad_token_id).sum().item()
                actual_prompt_len = prompt_length - num_pad
                print(f"  sample[{i}]: total={prompt_length}, pad_tokens={num_pad}, actual_prompt_len={actual_prompt_len}")
            # check multi_modal_data
            if "multi_modal_data" in prompts.non_tensor_batch:
                mm_data = prompts.non_tensor_batch["multi_modal_data"]
                print(f"  multi_modal_data keys: {list(mm_data.keys()) if isinstance(mm_data, dict) else type(mm_data)}")
                if isinstance(mm_data, dict) and "audio" in mm_data:
                    audios = mm_data["audio"]
                    print(f"  multi_modal_data audio: {len(audios)} entries, type: {type(audios[0])}")
            elif "multi_modal_inputs" in prompts.non_tensor_batch:
                mm = prompts.non_tensor_batch["multi_modal_inputs"]
                if isinstance(mm, dict):
                    print(f"  multi_modal_inputs keys: {list(mm.keys())}")
                    for k, v in mm.items():
                        if hasattr(v, 'shape'):
                            print(f"    {k}: shape={v.shape}")
                        elif isinstance(v, (list, tuple)):
                            print(f"    {k}: len={len(v)}, type={type(v[0]) if v else 'empty'}")
                        else:
                            print(f"    {k}: type={type(v)}")
            print(f"  max_prompt_length (config): {self.config.get('max_prompt_length', 'N/A')}")
            print(f"  max_response_length (config): {self.config.get('max_response_length', 'N/A')}")
            print(f"  truncation (config): {self.config.get('truncation', 'N/A')}")
            print("=============================================")

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        if _ECHO_DEBUG:
            print("=== ECHO_DEBUG: pre-assertion state ===")
            print(f"  seq.shape: {seq.shape}")
            print(f"  prompt_length: {prompt_length}")
            print(f"  response_length (config): {response_length}")
            print(f"  sequence_length (expected): {sequence_length}")
            print(f"  generated_batch_size: {generated_batch_size}")
            print(f"  delta_length: {delta_length} (positive=needed padding, negative=output longer than expected)")
            print(f"  seq min/max token_id: {seq.min().item()} / {seq.max().item()}")
            print(f"  pad_token_id: {pad_token_id}")
            print(f"  eos_token_id: {eos_token_id}")
            print(f"  output.sequences shape before potential pad: {output.sequences.shape}")
            if hasattr(output, 'scores') and output.scores is not None:
                print(f"  output.scores: {len(output.scores)} elements")
            # Check if output sequence contains BOS at start (position 0)
            prompt_start_tokens = seq[:, :5].tolist() if seq.size(1) >= 5 else seq.tolist()
            print(f"  first 5 tokens of seq: {prompt_start_tokens}")
            # Compare input_ids first/last to output seq first/last
            print(f"  input_ids first 5: {idx[0, :5].tolist() if idx.size(1) >= 5 else idx[0].tolist()}")
            print(f"  input_ids last 5: {idx[0, -5:].tolist()}")
            # Check per-sample generated lengths
            for i in range(seq.size(0)):
                # Find EOS position relative to prompt end
                response_part = seq[i, prompt_length:]
                eos_positions = (response_part == eos_token_id).nonzero(as_tuple=True)[0]
                actual_response_len = eos_positions[0].item() if len(eos_positions) > 0 else response_part.size(0)
                print(f"  sample[{i}]: total_seq_len={seq.size(1)}, actual_response_len={actual_response_len} (first_eos_pos)")
            print("========================================")

        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
