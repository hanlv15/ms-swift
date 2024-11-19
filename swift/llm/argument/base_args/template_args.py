# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.llm import TEMPLATE_MAPPING
from swift.utils import get_logger

logger = get_logger()


@dataclass
class TemplateArguments:
    """
    TemplateArguments class is a dataclass that holds various arguments related to template configuration and usage.

    Args:
        template (Optional[str]): Template identifier. Default is None, meaning use the template of the model_type.
        system (Optional[str]): Override the default system in the template. Default is None.
        max_length (Optional[int]): Maximum length for the template. Default is None.
        truncation_strategy (Literal): Strategy for truncating the template. Default is 'left'.
        tools_prompt (str): Override the default tools prompt in the template. Default is 'react_en'.
        max_pixels (Optional[int]): Maximum number of pixels for the template. Default is None.
        loss_scale (str): Loss scale for training. Default is 'default',
            meaning only calculate the loss of the response.
        sequence_parallel_size (int): Size of sequence parallelism. Default is 1.
    """
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(TEMPLATE_MAPPING.keys())}'})
    system: Optional[str] = None  # Override the default_system in the template.
    max_length: Optional[int] = None

    truncation_strategy: Literal['delete', 'left'] = 'left'
    tools_prompt: str = 'react_en'  # Override the default_tools_prompt in the template.
    max_pixels: Optional[int] = None
    # train
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1

    def __post_init__(self):
        if self.template is None:
            self.template = self.model_meta.template

        if self.max_length is None:
            self.max_length = self.model_info.max_model_len