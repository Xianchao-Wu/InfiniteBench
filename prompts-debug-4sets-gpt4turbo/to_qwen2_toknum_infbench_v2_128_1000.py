import tiktoken
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = "/workspace/asr/megatron.20240606/qwen"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-72B-Instruct", cache_dir=cache_dir)
#gpt128k_enc = tiktoken.encoding_for_model('gpt-4-turbo')
gpt128k_enc = tokenizer #tiktoken.encoding_for_model('gpt-4-turbo')

#adir = "/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-ret5/"

#flist = [adir + '/longbook_choice_eng/prompts_longbook_choice_eng_131022_template_gpt4_topn5.json', 
#        adir + '/longbook_qa_eng/prompts_longbook_qa_eng_131022_template_gpt4_topn5.json']

adir = "/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-4sets-gpt4turbo/"

import os

def get_files(akey="qwen2_72b_instruct.txt"):
    files = list()
    for atask in ["longbook_choice_eng", "longbook_sum_eng", "longbook_qa_eng", "longdialogue_qa_eng"]:
        asubdir = adir + atask
        for afile in os.listdir(asubdir):
            if 'cut' in afile or 'same' in afile: # or 'json' in afile:
                continue
            if akey in afile and not afile.endswith('.json'):
                afile = asubdir + '/' + afile
                files.append(afile)
    for afile in files:
        print('target file: ', afile)
    return files

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

def count(afile):
    all_len = 0
    numlines = 0
    predictions = load_out_sep_by_assistant(afile)
    all_len_list = list()

    for out in predictions:
        #import ipdb; ipdb.set_trace()
        alen = len(gpt128k_enc.encode(out))
        all_len_list.append(alen)
        #print(alen)
        all_len += alen
        numlines += 1
    avg = all_len / float(len(predictions))
    print(afile, numlines, all_len, avg)
    print(all_len_list)
    return all_len

#flist = get_files(akey="start")
flist = get_files()
for afile in flist:
    count(afile)


