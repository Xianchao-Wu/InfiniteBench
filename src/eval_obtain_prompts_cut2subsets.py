import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

import tiktoken

gpt4_tokenizer = tiktoken.encoding_for_model('gpt-4-turbo')

from eval_utils import (
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from args import parse_args

MAX_POSITION_ID = 128 * 1024  # Determined by the model
KEEP_LEN = 50 # TODO help
TRUNCATE_LEN = 128 * 1024 - KEEP_LEN


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:] # 中间的一些内容，不要了!!!保留最左边一半，以及最右边一半内容 NOTE
    else:
        return None

# input = input prompt text; tok = tokenizer; max_tokens = max num of tokens of input; tok_llama3 = llama3 tokenizer (for tok num counting only)
def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle",
        tok_llama3: AutoTokenizer = None):
    tokens = tok.encode(input)
    len_before = len(tokens) # num of tokens before pruning (truncating)

    #len_before_llama3 = len(tok_llama3.encode(input))
    #print(f"# tokens before: {len_before} {len_before_llama3} of llama3 tok")
    print(f"# tokens before: {len_before} ")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens

    # is_cut = True -> pruned, is_cut = False -> not pruned and kept NOTE
    is_cut = len_after < len_before

    return tok.decode(tokens), is_cut, len_after

def get_prompt(
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text, is_cut, cur_tok = truncate_by_tokens(input_text, tok, TRUNCATE_LEN) 
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    return input_text, is_cut, cur_tok # for prompt, to be used by existing open-source LLMs, such as qwen2-72b-instruct

def load_tokenizer(
    model_name: str = None, 
    cache_dir: str = None,
) -> AutoTokenizer:
    print("Loading tokenizer: {}".format(model_name))
    tok = AutoTokenizer.from_pretrained(model_name, 
            cache_dir=cache_dir)
    return tok  # type: ignore

if __name__ == "__main__":
    model_name = "yarn-mistral"
    args = parse_args()
    if args.model_name is not None and len(args.model_name) > 0:
        model_name = args.model_name
    #import ipdb; ipdb.set_trace()
    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Tokenizer
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    #tok = load_tokenizer(args.model_path, args.cache_dir)
    tok = gpt4_tokenizer # NOTE 这里使用的是gpt-4-turbo作为tokenizer来切割prompts, 128k

    # Data
    result_dir = Path(args.output_dir, data_name) # NOTE
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)
    
    #TRUNCATE_LEN = 128 * 1024 - KEEP_LEN
    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path_same = (
            result_dir / f"prompts_{data_name}_{TRUNCATE_LEN}_template_{model_name}_same.json"
        )
        output_path_cut = (
            result_dir / f"prompts_{data_name}_{TRUNCATE_LEN}_template_{model_name}_cut.json"
        )
    else:
        output_path_same = (
            result_dir / f"prompts_{data_name}_{args.start_idx}-{args.stop_idx}_{TRUNCATE_LEN}_same.json"  # noqa
        )
        output_path_cut = (
            result_dir / f"prompts_{data_name}_{args.start_idx}-{args.stop_idx}_{TRUNCATE_LEN}_cut.json"  # noqa
        )
    
    # prompts_same = not pruned NOTE
    prompts_same, prompts_cut = [], []
    num_same, num_cut, num_same_tok, num_cut_tok = 0, 0, 0, 0
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        #import ipdb; ipdb.set_trace()
        eg = examples[i] # a dict: dict_keys(['id', 'context', 'input', 'answer']) for 'longbook_qa_eng' one sample NOTE ||| dict_keys(['id', 'context', 'input', 'answer', 'options']) for 'longbook_choice_eng' one sample, eg['answer']=['Walking Georgie']; and eg['options']=['Walking Georgie', 'Taking care of Totty', 'Working in the dairy', 'Light housework']
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        #import ipdb; ipdb.set_trace()

        print(f"====== Example {i} ======")
        #import ipdb; ipdb.set_trace()
        prompt, is_cut, cur_tok = get_prompt(
            tok, input_text, max_tokens=max_tokens, verbose=args.verbose
        )
        print('cutinfo[idx>=0,is_cut,data_name]: {} {} {}'.format(i, is_cut, data_name))
        if args.verbose:
            print(prompt)
        prompts = prompts_cut if is_cut else prompts_same
        prompts.append(
            {
                "prompt": prompt,
            }
        )
        if is_cut:
            num_cut += 1
            num_cut_tok += cur_tok
        else:
            num_same += 1
            num_same_tok += cur_tok
    # finally, save prompts to output_path_same and output_path_cut
    # TODO
    with open(output_path_same, 'w') as f:
        json.dump(prompts_same, f, indent=2)
    
    with open(output_path_cut, 'w') as f:
        json.dump(prompts_cut, f, indent=2)
    
    avg_same, avg_cut = float(num_same_tok)/num_same, float(num_cut_tok)/num_cut
    print('done.final_display:[avg_same;num_same;avg_cut;num_cut]: {} {} {} {} {}'.format(
        avg_same, num_same, avg_cut, num_cut, data_name
        )
    )

