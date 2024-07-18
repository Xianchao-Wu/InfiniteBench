import tiktoken
import json

gpt128k_enc = tiktoken.encoding_for_model('gpt-4-turbo')

adir = "/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-ret5/"

flist = [adir + '/longbook_choice_eng/prompts_longbook_choice_eng_131022_template_gpt4_topn5.json', 
        adir + '/longbook_qa_eng/prompts_longbook_qa_eng_131022_template_gpt4_topn5.json']

def count(afile):
    outlines = list()
    all_len = 0
    numlines = 0
    with open(afile) as br:
        contents = br.read()
        contents_json = json.loads(contents)
        for asample in contents_json:
            #import ipdb; ipdb.set_trace()
            outlines.append(asample)
            prompt = asample['prompt']
            alen = len(gpt128k_enc.encode(prompt))
            #print(alen)
            all_len += alen
            numlines += 1
    print(afile, numlines, all_len)
    return all_len

for afile in flist:
    count(afile)


