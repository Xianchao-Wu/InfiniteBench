import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from yarn_mistral.modeling_mistral_yarn import MistralForCausalLM
from args import parse_args


MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 128 * 1024


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
    return tok.decode(tokens, skip_special_tokens=True)


def chunk_generate(
    model,
    tok,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = 128 * 1024,
    chunk_size: int = 2500,
    verbose: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    #debug = True
    #if debug:
    #    return ['debug only as ouptut']

    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", padding=True) # {'input_ids': tensor([[    1,  4939,   272,  ...,  2820, 16981, 28747]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]])}
        inputs = inputs.to(model.device)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n), e.g., (1, 94244)
        attention_mask: Tensor = inputs.attention_mask  # (b, n), e.g., (1, 94244)
        position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1 # torch.Size([1, 94244])
        position_ids.masked_fill_(attention_mask == 0, value=1)
        seq_len = input_ids.shape[-1]
        print("seq_len:", seq_len)
        kv_cache: Any = None
        # Split into chunks for pre-filling
        chunk_idxs = []
        n = seq_len - 1 # 94244-1=94243
        while n > 0:
            chunk_idxs.append(n)
            n -= chunk_size # 128=chunk_size
        chunk_idxs.append(0)
        chunk_idxs = chunk_idxs[::-1] # 整个数组，翻转一下，头变尾；尾变头
        chunk_lo = chunk_idxs[:-1]
        chunk_hi = chunk_idxs[1:]
        print(f"Number of chunks: {len(chunk_lo)}, generating...") # 737 chunks NOTE
        start_time = time.time()

        import ipdb; ipdb.set_trace()
        for chunk_i, (chunk_lo, chunk_hi) in enumerate(
            zip(chunk_lo, chunk_hi)
        ):
            if verbose:
                print(
                    f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
                    round(time.time() - start_time),
                )
            chunk_input_ids = input_ids[:, chunk_lo:chunk_hi] # tok seq
            if kv_cache is not None:
                mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
            else:
                mask_start_idx = chunk_lo
            chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi] # all 1, torch.Size([1, 35]); ||| all 1, [1, 163]
            chunk_position_ids = position_ids[:, chunk_lo:chunk_hi] # [1, 94244]=shape with elements from 0 to 94243: tensor([[    0,     1,     2,  ..., 94241, 94242, 94243]], device='cuda:0') ||| [1, 128] from 35 to 162

            #import ipdb; ipdb.set_trace()
            outputs: BaseModelOutputWithPast = model.model.forward(
                input_ids=chunk_input_ids, # tokens, [1, 35]=shape ||| [1, 128]
                attention_mask=chunk_attention_mask, # all 1, [1, 35]=shape ||| [1, 163]
                position_ids=chunk_position_ids, # from 0 to 34, [1, 35]=shape ||| from 35 to 162, [1, 128]=shape
                past_key_values=kv_cache, # None ||| len=32 layers
                return_dict=True,
                use_cache=True,
            ) # outputs[0].shape=[1, 35, 4096] ||| outputs[0].shape=[1, 128, 4096]
            kv_cache = outputs.past_key_values # a tuple with 32 elements
            # Discard KV states on the left beyond the window
            new_cache = ()
            n_layers = len(kv_cache) # 32
            for layer_i in range(n_layers):
                keys = kv_cache[layer_i][0][:, :, -sliding_window:] # sliding_window=131072; keys=torch.Size([1, 8, 35, 128])
                values = kv_cache[layer_i][1][:, :, -sliding_window:] # torch.Size([1, 8, 35, 128])
                new_cache += ((keys, values),)
            kv_cache = new_cache
        kv_cache_len = kv_cache[0][0].shape[2]
        
        import ipdb; ipdb.set_trace()
        outputs = model.generate(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask[:, -kv_cache_len - 1 :],
            max_new_tokens=max_tokens,
            past_key_values=kv_cache, # NOTE torch.Size([1, 8, 94243, 128])=kv_cache[0][0].shape
            eos_token_id=tok.pad_token_id,
            use_cache=True,
        )
        responses = [
            tok.decode(t[1:], skip_special_tokens=True) for t in outputs
        ]
    return responses


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    tok_llama3: AutoTokenizer = None
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN, 
            tok_llama3=tok_llama3)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    import ipdb; ipdb.set_trace()
    output = chunk_generate(
        model, # <class 'yarn_mistral.modeling_mistral_yarn.MistralForCausalLM'>
        tok, # <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
        [input_text],
        max_tokens=max_tokens, # 40 NOTE this is max output tokens
        chunk_size=128,
        verbose=verbose, # False
    )[0]
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
) -> Tuple[MistralForCausalLM, AutoTokenizer]:
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name, 
            cache_dir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/cache/")
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    #import ipdb; ipdb.set_trace()
    model = MistralForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
        cache_dir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/cache/"
    )
    print("Time taken:", round(time.time() - start_time))
    return model, tok  # type: ignore


if __name__ == "__main__":
    model_name = "yarn-mistral"
    args = parse_args()
    import ipdb; ipdb.set_trace()
    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)

    tok_llama3 = AutoTokenizer.from_pretrained(
            'meta-llama/Meta-Llama-3-8B', 
            cache_dir=".")

    # Data
    result_dir = Path(args.output_dir, model_name) # PosixPath('../results/yarn-mistral') NOTE
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i] # a dict: dict_keys(['id', 'context', 'input', 'answer']) for 'longbook_qa_eng' one sample NOTE ||| dict_keys(['id', 'context', 'input', 'answer', 'options']) for 'longbook_choice_eng' one sample, eg['answer']=['Walking Georgie']; and eg['options']=['Walking Georgie', 'Taking care of Totty', 'Working in the dairy', 'Light housework']
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        import ipdb; ipdb.set_trace()
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose,
            tok_llama3=tok_llama3
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
