# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import json
from transformers.utils.versions import require_version

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelInfo, PtLoRARequest, get_template_meta
from swift.utils import get_logger, is_lmdeploy_available, is_vllm_available
from .base_args import BaseArguments, to_abspath
from .lmdeploy_args import LmdeployArguments
from .merge_args import MergeArguments
from .tuner_args import adapters_can_be_merged
from .vllm_args import VllmArguments

logger = get_logger()


@dataclass
class InferArguments(BaseArguments, MergeArguments, VllmArguments, LmdeployArguments):
    infer_backend: Literal['vllm', 'pt', 'lmdeploy'] = 'pt'
    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})

    val_dataset_sample: int = -1
    result_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/infer_result'})
    save_result: bool = True

    # other
    stream: Optional[bool] = None

    def _init_result_dir(self) -> None:
        self.result_path = None
        if not self.save_result:
            return

        if self.result_dir is None:
            if self.ckpt_dir is None:
                result_dir = self.model_info.model_dir
            else:
                result_dir = self.ckpt_dir
            result_dir = os.path.join(result_dir, 'infer_result')
        else:
            result_dir = self.result_dir
        result_dir = to_abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.result_path = os.path.join(result_dir, f'{time}.jsonl')
        logger.info(f'args.result_path: {self.result_path}')

    def _init_stream(self):
        self.eval_human = not (self.dataset and self.split_dataset_ratio > 0 or self.val_dataset)
        if self.stream is None:
            self.stream = self.eval_human

        template_meta = get_template_meta(self.template)
        if self.num_beams != 1 or not template_meta.support_stream:
            self.stream = False
            logger.info('Setting args.stream: False')

    def __post_init__(self) -> None:
        BaseArguments.__post_init__(self)
        MergeArguments.__post_init__(self)
        VllmArguments.__post_init__(self)

        self._init_result_dir()
        self._init_stream()
        self._init_eval_human()
        self._parse_lora_modules()

    def _init_eval_human(self):
        if len(self.dataset) == 0 and len(self.val_dataset) == 0:
            eval_human = True
        else:
            eval_human = False
        self.eval_human = eval_human
        logger.info(f'Setting args.eval_human: {self.eval_human}')

    def _parse_lora_modules(self) -> None:
        if len(self.lora_modules) == 0:
            return
        assert self.infer_backend in {'vllm', 'pt'}
        if self.infer_backend == 'vllm':
            from vllm.lora.request import LoRARequest
            lora_request_cls = LoRARequest
        elif self.infer_backend == 'pt':
            lora_request_cls = PtLoRARequest

        lora_request_list = []
        for i, lora_module in enumerate(self.lora_modules):
            lora_name, lora_local_path = lora_module.split('=')
            lora_request_list.append(lora_request_cls(lora_name, i + 1, lora_local_path))
        self.lora_request_list = lora_request_list