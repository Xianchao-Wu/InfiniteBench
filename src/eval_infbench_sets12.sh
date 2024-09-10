#########################################################################
# File Name: eval_infbench_sets12.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Sep 10 08:14:18 2024
#########################################################################
#!/bin/bash

if [[ $# -lt 3 ]]; then
	echo "Usage: $0 <task_name> <test_output_file> <reference_file>"
	echo "    task_name: passkey number_string kv_retrieval longbook_sum_eng longbook_qa_eng longbook_qa_chn longbook_choice_eng longdialogue_qa_eng math_calc math_find code_debug code_run"
	echo "    test_output_file: the path of the llm's output file"
	echo "    reference_file: the reference jsonl file downloaded from https://github.com/OpenBMB/InfiniteBench/tree/main?tab=readme-ov-file#using-scripts"
	exit 1
fi

task_name=$1
pxout_txt=$2
pxref_json=$3

#datetime=$(date +%Y%m%d-%H%M%S)
datetime=$(date +%Y%m%d)
#echo ${datetime}

#sets7="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240709/long_131072_25_multiturn_qa_blend_continue_pp1/7sets"
#sets7="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240710/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_pp1/7sets/"
#sets7="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240711/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_pp1/7sets"

#sets12="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-12sets"
#sets12="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240715/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_qa_blend_llama3_books_131072_65536_70b_64_3e-7_step_3300_pp1"

#rootdir="/workspace/asr/megatron.20240606/infbench-20240609/"

#refdir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/data"

#logfn="${rootdir}/InfiniteBench/src/3.prompts_cut2subsets.sh.log"

#indir="$rootdir/InfiniteBench/prompts-debug-ret5"

# TODO works for both "longbook_qa_eng" and "longbook_choice_eng"
function longbook_eng_eval(){
	task_name=$1 
	pxout_txt=$2
	pxref_json=$3	

	pxout_ref_json="${pxout_txt}.${datetime}.json"

	python prepare_json_from_pxout.py \
		--task ${task_name} \
		--pxout_txt ${pxout_txt} \
		--pxref_json ${pxref_json} \
		--pxout_ref_json ${pxout_ref_json} \
		--sep_by_assistant

	if [[ $task_name =~ "longbook_choice" ]]
	then
		python compute_scores_pengxu_2sets_20240609.py \
			--task ${task_name} \
			--pxout_ref_json ${pxout_ref_json} \
			--model_name pxlong \
			--use_zero_scrolls
	fi

	python compute_scores_pengxu_2sets_20240609.py \
		--task ${task_name} \
		--pxout_ref_json ${pxout_ref_json} \
		--model_name pxlong
}

longbook_eng_eval ${task_name} ${pxout_txt} ${pxref_json}

