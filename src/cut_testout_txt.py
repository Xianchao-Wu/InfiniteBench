import os
import sys
import json

logfn=sys.argv[1]
testout_txt_fn=sys.argv[2]
data_name=sys.argv[3]

print(logfn)
print(testout_txt_fn)

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

def is_sep_by_assistant(testout_txt_fn):
    out_flag = False
    with open(testout_txt_fn) as br:
        for aline in br.readlines():
            if aline.startswith('assistant: '):
                out_flag = True
                break
    return out_flag

def load_testout_longbook_qa_eng(testout_txt_fn):
    is_sep_by_assistant_flag = is_sep_by_assistant(testout_txt_fn)
    if is_sep_by_assistant_flag:
        return load_out_sep_by_assistant(testout_txt_fn)

    # NOTE normal separate by line
    testouts = list()
    with open(testout_txt_fn) as br:
        for aline in br.readlines():
            aline = aline.strip()
            testouts.append(aline)
    return testouts

def load_out_sep_by_assistant(testout_txt_fn):
    # NOTE separate by 'assistant: '
    predictions = list()
    with open(testout_txt_fn, 'r') as f:
        one_answer = ''
        for line in f.readlines():
            line = line.strip()
            if line.startswith('assistant: '):
                if len(one_answer) > 0:
                    predictions.append(one_answer)
                    one_answer = ''
                #one_answer = line[len('assistant: '):]
                one_answer = line #[len('assistant: '):]
            else:
                one_answer += '\n' + line
        if len(one_answer) > 0:
            predictions.append(one_answer)
    return predictions


def save_to_txt(data, fname):
    print('save to txt: {} samples to {}'.format(len(data), fname))
    with open(fname, 'w', encoding='utf8') as fout:
        for asample in data:
            fout.write(asample)
            fout.write('\n')
    print('saved to txt: {} samples to {}'.format(len(data), fname))

cutdict = load_log(logfn, data_name)
testoutlist = load_testout_longbook_qa_eng(testout_txt_fn)

set_same, set_cut = list(), list()
for i in range(len(cutdict)):
    flag = cutdict[str(i)]
    print(i, flag)

    if flag:
        set_same.append(testoutlist[i])
    else:
        set_cut.append(testoutlist[i])

# save two lists to files
same_fn = testout_txt_fn + ".same" 
cut_fn = testout_txt_fn + ".cut"

save_to_txt(set_same, same_fn)
save_to_txt(set_cut, cut_fn)


