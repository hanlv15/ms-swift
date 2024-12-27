# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import (InferRequest, Model, ModelGroup, ModelMeta, PtEngine, RequestConfig, TemplateMeta,
                       get_model_tokenizer_with_flash_attn, register_model, register_template)

from swift.llm.template.constant import LLMTemplateType
from swift.llm.model.constant import LLMModelType
from swift.llm.model.model_arch import ModelArch

class CustomModelType:
    tigerbot_7b = 'tigerbot-7b'
    tigerbot_13b = 'tigerbot-13b'
    tigerbot_13b_chat = 'tigerbot-13b-chat'

    orca2_7b = "orca-2-13b"
    openchat_35 = "openchat_3.5"
    fusechat_7b = "fusechat-7b-varm"
    mistral_7b_instruct_v3 = "mistral-7b-instruct-v0.3"
    neural_chat_7b = "neural-chat-7b-v3"
    solar_instruct_10_7b = "solar-10.7b-instruct"
    mixtral_moe_7b_instruct_gptq_int4 = "mixtral-8x7B-instruct-v0.1-gptq-int4"
    mixtral_moe_7b_instruct_awq = "mixtral-8x7B-instruct-v0.1-awq"
    gemma_2_9b_it = 'gemma-2-9b-it'
    merlinite_7b = 'merlinite-7b'
    c4ai_command_r_4bit = "c4ai-command-r-v01-4bit"
    llama_3_8b_instruct = "meta-llama-3-8B-instruct"
    llama_3_70b_instruct_gptq_int4 = "meta-llama3-70B-instruct-gptq-int4"
    llama_3_70b_instruct_awq = "meta-llama-3-70b-instruct-awq"

    phi_3_mini_4k_instruct = "phi-3-mini-4k-instruct"
    phi_3_small_8k_instruct = "phi-3-small-8k-instruct"
    phi_3_medium_4k_instruct = "phi-3-medium-4k-instruct"

class CustomTemplateType:
    tigerbot = 'tigerbot'

    orca2 = "orca-2"
    openchat_35 = "openchat_3.5"
    neural = "neural"
    solar = "solar"
    # mistral = "mistral"
    chatml = "_chatml" # 无system message的chatml
    llama = "_llama" # 无system message的llama
    merlinite = "merlinite"
    c4ai_command_r = "c4ai_command_r" # 用于RAG的Template
    llama3 = "_llama3"
    gemma = '_gemma'
    phi3 = '_phi3'

register_template(
    TemplateMeta(
        template_type='custom',
        prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
        prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
        chat_sep=['\n']))

# register_template(
#     TemplateMeta(
#         template_type=CustomTemplateType.llama3,
#         prefix=['<|begin_of_text|>'],
#         prompt=[
#         '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
#         '<|start_header_id|>assistant<|end_header_id|>\n\n'
#     ],
#         chat_sep=['<|eot_id|>'],
#         suffix=['<|eot_id|>'],
#         default_system=None,
#         system_prefix=['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>']
# ))

################################

register_model(
    ModelMeta(
        model_type='custom',
        model_groups=[
            ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct', 'nvidia/Nemotron-Mini-4B-Instruct')])
        ],
        template='custom',
        get_function=get_model_tokenizer_with_flash_attn,
        ignore_patterns=['nemo']))

register_model(
    ModelMeta(
        model_type=LLMModelType.llama3,
        model_groups=[
            ModelGroup([
                Model(model_path='/home/css/models/llama-3-70b-instruct-awq'),
            ])
        ],
        template=LLMTemplateType.llama3,
        get_function=get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        model_type=LLMModelType.llama3_2,
        model_groups=[
            ModelGroup([
                Model(model_path='/home/css/models/llama-3.3-70b-instruct-awq'),
            ])
        ],
        template=LLMTemplateType.llama3_2,
        get_function=get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        requires=['transformers>=4.45'],
        model_arch=ModelArch.llama,
    ))

if __name__ == '__main__':
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
    request_config = RequestConfig(max_tokens=512, temperature=0)
    engine = PtEngine('AI-ModelScope/Nemotron-Mini-4B-Instruct')
    response = engine.infer([infer_request], request_config)
    swift_response = response[0].choices[0].message.content

    engine.default_template.template_backend = 'jinja'
    response = engine.infer([infer_request], request_config)
    jinja_response = response[0].choices[0].message.content
    assert swift_response == jinja_response, (f'swift_response: {swift_response}\njinja_response: {jinja_response}')
    print(f'response: {swift_response}')
