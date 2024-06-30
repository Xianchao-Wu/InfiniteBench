import os
import sys
import json

logfn=sys.argv[1]
pxref_json_fn=sys.argv[2]
data_name=sys.argv[3]

print(logfn)
print(pxref_json_fn)

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

def load_ref_longbook_qa_eng(pxref_json_fn):
    refs = list()
    with open(pxref_json_fn) as br:
        file_contents = br.read()
        file_contents_json = json.loads(file_contents)
        for asample in file_contents_json:
            #ref = asample['answers'] # NOTE keep this as a list!
            #refs.append(ref)
            refs.append(asample)
    return refs

def dump_json(data, fname):
    print('dump json: {} samples to {}'.format(len(data), fname))
    with open(fname, 'w', encoding='utf8') as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

cutdict = load_log(logfn, data_name)
reflist = load_ref_longbook_qa_eng(pxref_json_fn)

set_same, set_cut = list(), list()
for i in range(len(cutdict)):
    flag = cutdict[str(i)]
    print(i, flag)

    if flag:
        set_same.append(reflist[i])
    else:
        set_cut.append(reflist[i])

# save two lists to files
same_fn = pxref_json_fn.replace('test.json', 'test_same.json')
cut_fn = pxref_json_fn.replace('test.json', 'test_cut.json')

dump_json(set_same, same_fn)
dump_json(set_cut, cut_fn)






