import os.path
from functools import partial
from typing import Type

import gradio as gr

from swift.llm import MODEL_MAPPING, ModelType
from swift.llm.model.register import get_all_models
from swift.ui.base import BaseUI


class Model(BaseUI):

    group = 'llm_eval'

    locale_dict = {
        'checkpoint': {
            'value': {
                'zh': '训练后的模型',
                'en': 'Trained model'
            }
        },
        'model_type': {
            'label': {
                'zh': '选择模型类型',
                'en': 'Select Model Type'
            },
            'info': {
                'zh': 'SWIFT已支持的模型类型',
                'en': 'Base model type supported by SWIFT'
            }
        },
        'model': {
            'label': {
                'zh': '模型id或路径',
                'en': 'Model id or path'
            },
            'info': {
                'zh': '实际的模型id，如果是训练后的模型请填入checkpoint-xxx的目录',
                'en': 'The actual model id or path, if is a trained model, please fill in the checkpoint-xxx dir'
            }
        },
        'reset': {
            'value': {
                'zh': '恢复初始值',
                'en': 'Reset to default'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            model = gr.Dropdown(elem_id='model', scale=20, choices=get_all_models(), allow_custom_value=True)
            model_type = gr.Dropdown(elem_id='model_type', choices=ModelType.get_model_name_list(), scale=20)

    @classmethod
    def after_build_ui(cls, base_tab: Type['BaseUI']):
        cls.element('model').change(
            partial(cls.update_input_model, has_record=False),
            inputs=[cls.element('model')],
            outputs=list(cls.valid_elements().values()))
