# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from swift.llm import HfConfigFactory


@dataclass
class QuantizeArguments:
    # awq, gptq, and aqlm need to be pre-quantized models,
    # while bnb, hqq, and eetq can be quantized during SFT using the original models.
    quant_method: Literal['bnb', 'hqq', 'eetq', 'awq', 'gptq'] = None
    # bnb: 4,8; hqq: 1,2,3,4,8'; eetq: 8
    # awq: 4; gptq: 2,3,4,8
    quant_bits: Literal[0, 1, 2, 3, 4, 8] = 0
    # hqq
    hqq_axis: Optional[int] = None
    # bnb
    bnb_4bit_compute_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True

    def _init_quantization_config(self) -> None:
        if self.quant_method is None:
            self.quantization_config = None
            return
        assert self.quant_method in {'bnb', 'hqq', 'eetq'}
        if self.quant_method == 'bnb':
            if self.quant_bits == 4:
                load_in_4bit, load_in_8bit = True, False
            elif self.quant_bits == 8:
                load_in_4bit, load_in_8bit = False, True

            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant)
        elif self.quant_method == 'hqq':
            from transformers import HqqConfig
            quantization_config = HqqConfig(nbits=self.quant_bits, axis=self.hqq_axis)
        else:  # 'eetq'
            from transformers import EetqConfig
            quantization_config = EetqConfig(f'int{self.quant_bits}')

        self.quantization_config = quantization_config

    def __post_init__(self):
        from swift.llm import ExportArguments
        if self.bnb_4bit_compute_dtype is None:
            if self.torch_dtype in {torch.float16, torch.float32}:
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.torch_dtype == torch.bfloat16:
                self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_compute_dtype: torch.dtype = HfConfigFactory.to_torch_dtype(self.bnb_4bit_compute_dtype)
        if not isinstance(self, ExportArguments):
            self._init_quantization_config()
