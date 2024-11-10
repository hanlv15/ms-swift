from swift.llm import TemplateInputs, get_model_tokenizer, get_template, load_dataset


def test_template():
    _, tokenizer = get_model_tokenizer('qwen/Qwen2-7B-Instruct', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template_inputs = TemplateInputs([{
        'role': 'system',
        'content': 'AAA'
    }, {
        'role': 'user',
        'content': 'BBB'
    }, {
        'role': 'assistant',
        'content': 'CCC'
    }, {
        'role': 'user',
        'content': 'DDD'
    }])
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(tokenizer.decode(inputs['input_ids']))


def test_mllm():
    _, tokenizer = get_model_tokenizer('qwen/Qwen2-VL-7B-Instruct', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template_inputs = TemplateInputs([{
        'role': 'system',
        'content': 'AAA'
    }, {
        'role': 'user',
        'content': '<image>BBB'
    }, {
        'role': 'assistant',
        'content': 'CCC'
    }, {
        'role': 'user',
        'content': 'DDD'
    }],
                                     images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'])
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(template.safe_decode(inputs['input_ids']))


def _test_dataset_map(model_id: str, dataset_id: str):
    _, tokenizer = get_model_tokenizer(model_id, load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    dataset = load_dataset([dataset_id])[0]
    from swift.llm.dataset.utils import EncodePreprocessor

    new_dataset = EncodePreprocessor(template)(dataset)
    print(template.safe_decode(new_dataset[0]['input_ids']))
    print(template.safe_decode(new_dataset[1]['input_ids']))


def test_llm_dataset_map():
    _test_dataset_map('qwen/Qwen2-7B-Instruct', 'AI-ModelScope/alpaca-gpt4-data-zh#1000')


def test_mllm_dataset_map():
    _test_dataset_map('qwen/Qwen2-VL-7B-Instruct', 'modelscope/coco_2014_caption:validation')


if __name__ == '__main__':
    # test_template()
    # test_mllm()
    test_llm_dataset_map()
    test_mllm_dataset_map()
