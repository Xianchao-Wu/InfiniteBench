import os
import sys
import json

if len(sys.argv) < 4:
    print("Usage: {} <logfn> <testout_json_fn> <data_name>".format(sys.argv[0]))
    os._exit(1)

logfn=sys.argv[1]
testout_json_fn=sys.argv[2]
data_name=sys.argv[3]

print(logfn)
print(testout_json_fn)

def load_log(logfn, data_name):
    outdict = dict()
    with open(logfn) as br:
        for aline in br.readlines():
            aline = aline.strip()
            if aline.startswith('cutinfo') and data_name in aline:
                # cutinfo[idx>=0,is_cut,data_name]: 0 False longbook_qa_eng
                cols = aline.split(' ')
                aid = cols[1]
                aflag = True if cols[2] == 'True' else False
                outdict[aid] = aflag
    return outdict

def load_testout_official_json(testout_json_fn):
    # NOTE normal separate by line
    testouts = list()
    with open(testout_json_fn) as br:
        for aline in br.readlines():
            aline = aline.strip()
            testouts.append(aline)
    return testouts

def save_to_jsonl(data, fname):
    print('save to jsonl: {} samples to {}'.format(len(data), fname))
    with open(fname, 'w', encoding='utf8') as fout:
        for asample in data:
            fout.write(asample)
            fout.write('\n')
    print('saved to jsonl: {} samples to {}'.format(len(data), fname))

cutdict = load_log(logfn, data_name)
testoutlist = load_testout_official_json(testout_json_fn)

set_same, set_cut = list(), list()
for i in range(len(cutdict)):
    flag = cutdict[str(i)]
    print(i, flag)

    if flag:
        set_same.append(testoutlist[i])
    else:
        set_cut.append(testoutlist[i])

# save two lists to files
same_fn = testout_json_fn + ".same.json" 
cut_fn = testout_json_fn + ".cut.json"

save_to_jsonl(set_same, same_fn)
save_to_jsonl(set_cut, cut_fn)


