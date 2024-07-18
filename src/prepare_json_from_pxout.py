from pathlib import Path
import json
import time
from datetime import datetime

from args import parse_args
from eval_utils import dump_jsonl, get_answer

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

def load_out(pxout_txt_fn):
    outs = list()
    with open(pxout_txt_fn) as br:
        for aline in br.readlines():
            aline = aline.strip()
            outs.append(aline)
    return outs

def load_ref_longbook_qa_eng(pxref_json_fn):
    refs = list()
    with open(pxref_json_fn) as br:
        file_contents = br.read()
        file_contents_json = json.loads(file_contents)
        for asample in file_contents_json:
            ref = asample['answers'] # NOTE keep this as a list!
            refs.append(ref)
    return refs

def load_ref_longbook_choice_eng(pxref_json_fn, task):
    refs = list()
    with open(pxref_json_fn) as br:
        file_contents = br.read()
        #import ipdb; ipdb.set_trace()
        file_contents_json = json.loads(file_contents)
        #import ipdb; ipdb.set_trace()
        for asample in file_contents_json:
            #ref = asample['answers'] # NOTE keep this as a list!
            #refs.append(ref)
            asample['options'] = asample['multichoice_options']
            asample['answer'] = asample['answers']
            #import ipdb; ipdb.set_trace()
            ref = get_answer(asample, task) 
            refs.append(ref)
    return refs

def load_ref_sets7(ref_jsonl_fn):
    refs = list()
    with open(ref_jsonl_fn, 'r') as br:
        for aline in br.readlines():
            ref = json.loads(aline)
            ref_ans = ref['answer']
            refs.append(ref_ans)
    return refs

def load_ref(pxref_json_fn, task):
    if task == 'longbook_qa_eng':
        return load_ref_longbook_qa_eng(pxref_json_fn)
    elif task == 'longbook_choice_eng':
        return load_ref_longbook_choice_eng(pxref_json_fn, task)
    else:
        #raise ValueError("task={} not supported yet.".format(task))
        return load_ref_sets7(pxref_json_fn)

def combine_to_infb(outs, refs, output_path):
    #import ipdb; ipdb.set_trace()
    max_len = min(len(outs), len(refs))
    if len(outs) < len(refs):
        print("Warning: {} lines in prediction, less than {} lines in ref".format(len(outs), len(refs)))
        refs = refs[:max_len]
    if len(refs) < len(outs):
        print("Warning: {} lines in prediction, larger than {} lines in ref".format(len(outs), len(refs)))
        outs = outs[:max_len]

    preds = list()
    for i in range(0, max_len):
        preds.append(
            {
                "id": i,
                "prediction" : outs[i],
                "ground_truth" : refs[i], # TODO must be a list, not a str!
            }
        )
    dump_jsonl(preds, output_path)
    print('done. saved id-pred-ref to {}'.format(output_path))

def is_sep_by_assistant(testout_txt_fn):
    out_flag = False
    with open(testout_txt_fn) as br:
        for aline in br.readlines():
            if aline.startswith('assistant: '):
                out_flag = True
                break
    return out_flag

ALL_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longdialogue_qa_eng",
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "math_calc",
    "code_run",
    "code_debug",
]

if __name__ == "__main__":
    args = parse_args()
    # args.task for task name in ALL_TASKS
    # args.pxout_txt for predicted output file
    # args.pxref_json for test.json reference file
    
    if args.task is None or args.task not in ALL_TASKS:
        raise('Error: task name [{}] is None or not in {}'.format(args.task, ALL_TASKS))

    if args.pxout_txt is None or not Path(args.pxout_txt).exists():
        raise('Error: system prediction file [{}] is None or not exists.'.format(args.pxout_txt))
    
    if args.pxref_json is None or not Path(args.pxref_json).exists():
        raise('Error: system reference file [{}] is None or not exists.'.format(args.pxref_json))

    #import ipdb; ipdb.set_trace()
    if args.sep_by_assistant and is_sep_by_assistant(args.pxout_txt):
        outs = load_out_sep_by_assistant(args.pxout_txt)
    else:
        outs = load_out(args.pxout_txt)

    #import ipdb; ipdb.set_trace()
    refs = load_ref(args.pxref_json, args.task)

    # determine the output json file name:
    if args.pxout_ref_json is None: 
        flag = str(datetime.now()).replace(' ', '-').replace(':', '-')
        output_path = args.pxout_txt + '.' + flag + '.json'
    else:
        output_path = args.pxout_ref_json

    print('combine tst.out and ref, output to file: {}'.format(output_path))

    # combine tst.out and ref to <ref, test.out> for next step scoring:
    infb_json_fn = combine_to_infb(outs, refs, output_path)



