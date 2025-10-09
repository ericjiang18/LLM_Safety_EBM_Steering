import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Gemma 拒绝 token 示例（可根据实际情况调整）
GEMMA_REFUSAL_TOKS = [32000]  # 这里仅为示例，实际可根据模型 tokenizer 设定

def format_instruction_gemma_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    """
    按照 Gemma 官方文档格式化 prompt。
    Gemma 的 prompt 结构如下：
    <bos><start_of_turn>user
    用户输入<end_of_turn>
    <start_of_turn>model
    （模型输出）
    <end_of_turn>
    """

    # <bos> 通常是 tokenizer.bos_token 或者特殊 token id
    bos = tokenizer.bos_token or ""
    start_of_turn = "<start_of_turn>"
    end_of_turn = "<end_of_turn>"

    # 构建 prompt
    prompt = bos
    if system is not None:
        # Gemma 官方文档没有 system 字段，但如有可加在 user 前
        prompt += f"{start_of_turn}system\n{system}{end_of_turn}\n"

    prompt += f"{start_of_turn}user\n{instruction}{end_of_turn}\n"
    prompt += f"{start_of_turn}model\n"

    if output is not None:
        prompt += f"{output}{end_of_turn}\n"

    if not include_trailing_whitespace:
        prompt = prompt.rstrip()

    return prompt

def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True
):
    """
    批量格式化并分词
    """
    if outputs is not None:
        prompts = [
            format_instruction_gemma_chat(tokenizer=tokenizer, instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma_chat(tokenizer=tokenizer, instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_gemma_weights(model, direction: Float[Tensor, "d_model"]):
    """
    权重正交化，结构与 qwen_model.py 类似
    """
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_gemma_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    """
    激活加权，结构与 qwen_model.py 类似
    """
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class GemmaModel(ModelBase):
    """
    Gemma 模型适配类，继承自 ModelBase
    """

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # 设置 padding token 和方向
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # Gemma 的 assistant 开始 token，通常是 <start_of_turn>model
        return self.tokenizer.encode("<start_of_turn>model", add_special_tokens=False)

    def _get_refusal_toks(self):
        return GEMMA_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_gemma_weights, direction=direction, coeff=coeff, layer=layer)

    def formatter(self, instruction: str, system: str = None, history=None):
        """
        格式化输入，兼容 vLLM
        """
        return format_instruction_gemma_chat(
            tokenizer=self.tokenizer,
            instruction=instruction,
            system=system,
            include_trailing_whitespace=True
        )