import tiktoken
import json

gpt128k_enc = tiktoken.encoding_for_model('gpt-4-turbo')

num_samples = [200, 1726, 2000, 2000]
sets = ['qmsum', 'qasper', 'narrative_qa', 'quality']
types = ['full', 'ret5']

adir="/workspace/asr/iclr2023_eval_data/sa_rotation_20240529/0529-wed/to.openai.zero.scrolls"


def count(afile, num):
    outlines = list()
    all_len = 0
    numlines = 0
    with open(afile) as br:
        contents = br.read()
        contents_json = json.loads(contents)
        for asample in contents_json[:num]:
            #import ipdb; ipdb.set_trace()
            outlines.append(asample)
            prompt = asample['prompt']
            alen = len(gpt128k_enc.encode(prompt))
            #print(alen)
            all_len += alen
            numlines += 1
    print(afile, numlines, all_len)
    return all_len

for idx, adata in enumerate(sets):
    num = num_samples[idx]
    for atype in types:
        if adata == 'narrative_qa' and atype == 'full':
            maxlen = 127980
        else:
            maxlen = 131050

        afile = adir + '/' + "{}.2openai.{}_{}.omitmsg.rmspaces.longbenchins.json".format(
            adata, maxlen, atype
        )
        print(afile)
        count(afile, num)


