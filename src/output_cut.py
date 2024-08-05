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
outfn = sys.argv[2]

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

all_len, all_len2 = 0, 0

keep_len = 1024
with open(outfn, 'w') as bw:
    for apred in predictions:
        alen = len(tokenizer.encode(apred))
        all_len += alen

        apred_out = tokenizer.decode(tokenizer.encode(apred)[:keep_len])
        alen2 = len(tokenizer.encode(apred_out))
        all_len2 += alen2

        print(alen, all_len, alen2, all_len2)

        bw.write('assistant: ' + apred_out)
        bw.write('\n')
        bw.flush()

    avg_len = all_len/float(len(predictions))
    avg_len2 = all_len2/float(len(predictions))

print(avg_len, avg_len2)
