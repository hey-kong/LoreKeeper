import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a query and a document, determine whether the document correctly and sufficiently answers the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                     query=query, doc=doc)
    return output


def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 32768

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = 'Given a query and a document, determine whether the document correctly and sufficiently answers the query'

queries = ["What is TiDB",
           "Please introduce TiDB",
           "What is TiDB used for",
           "How good is TiDB",
           "Does TiDB support FOREIGN KEY"]

answer = ("TiDB is an open-source, distributed SQL database designed to handle Hybrid Transactional and Analytical Processing (HTAP) workloads. "
           "It is fully compatible with MySQL, offering horizontal scalability, strong consistency, and high availability. "
           "TiDB aims to provide a comprehensive database solution that supports Online Transactional Processing (OLTP), Online Analytical Processing (OLAP), and HTAP services, making it suitable for use cases requiring high availability and strong consistency with large-scale data.")

pairs = [format_instruction(task, query, answer) for query in queries]

# Tokenize the input texts
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
