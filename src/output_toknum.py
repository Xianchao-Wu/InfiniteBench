import sys
import os
import json

from transformers import AutoTokenizer

cache_dir="/workspace/asr/megatron.20240606/qwen"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-72B-Instruct", 
    cache_dir=cache_dir
)

afile = sys.argv[1]

def load_out_sep_by_assistant(pxout_txt_fn):
    predictions = list()
    with open(pxout_txt_fn, 'r') as f:
        one_answer = ''
        for line in f.readlines():
            line = line.strip()
            if line.startswith('assistant: '):
                if len(one_answer) > 0:
                    predictions.append(one_answer)
                    one_answer = ''
                one_answer = line[len('assistant: '):]
            else:
                one_answer += '\n' + line
        if len(one_answer) > 0:
            predictions.append(one_answer)
    return predictions

predictions = load_out_sep_by_assistant(afile)

all_len = 0

for apred in predictions:
    alen = len(tokenizer.encode(apred))
    all_len += alen
    print(alen, all_len)

avg_len = all_len/float(len(predictions))

print(avg_len)
