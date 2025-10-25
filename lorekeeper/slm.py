import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN_PROMPT_PREFIX = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
"""

LLAMA_PROMPT_PREFIX = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""


def get_model_type(model_name):
    model_lower = model_name.lower()
    if 'qwen3' in model_lower:
        return 'qwen3'
    elif 'llama' in model_lower:
        return 'llama'
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def _build_suffix(model_type):
    suffix_map = {
        'qwen3': (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n\n"
        ),
        'llama': (
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    }
    return suffix_map[model_type]

def query_prompt(chunk_list, query, model_name):
    model_type = get_model_type(model_name)
    prefix = QWEN_PROMPT_PREFIX if model_type == 'qwen3' else LLAMA_PROMPT_PREFIX
    chunks = "\n\n".join(chunk_list)

    return (
        f"{prefix}{chunks}\n\n"
        f"Given the above context, answer the question: {query}\n\n"
        f"Only give me the answer and do not output any other words."
        f"{_build_suffix(model_type)}"
    )

class CustomModelWrapper:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.model.config.eos_token_id
        self.model_type = get_model_type(model_path)

    def generate_answer(self, query, nodes):
        chunk_list = [node.text for node in nodes]
        prompt = query_prompt(chunk_list, query, self.model_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
            )
        generated_ids = outputs[0]
        input_length = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        return answer
