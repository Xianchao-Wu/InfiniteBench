#########################################################################
# File Name: 1.comb.tstout.ref.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 08:14:18 2024
#########################################################################
#!/bin/bash

datetime=$(date +%Y%m%d-%H%M%S)
echo ${datetime}

# good for 'choice' dataset
test1=1
if [ $test1 -eq 1 ]
then
	task_name="longbook_choice_eng"
	adir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug/longbook_choice_eng"
	fn1="$adir/prompts_longbook_choice_eng_131022_template_claude2.json.qwen2_72b_instruct.txt"
	fn2="$adir/prompts_longbook_choice_eng_131022_template_gpt4.json.qwen2_72b_instruct.txt"

	fn3="$adir/prompts_longbook_choice_eng_131022_template_kimi.json.qwen2_72b_instruct.txt"
	fn4="$adir/prompts_longbook_choice_eng_131022_template_yarn-mistral.json.qwen2_72b_instruct.txt"

	for pxout_txt in $fn1 $fn2 $fn3 $fn4 
	do
		pxref_json="../../longbook_choice/test.json"
		pxout_ref_json="${pxout_txt}.${datetime}.json"
		python prepare_json_from_pxout.py --task ${task_name} \
			--pxout_txt ${pxout_txt} \
			--pxref_json ${pxref_json} \
			--pxout_ref_json ${pxout_ref_json} \
			--sep_by_assistant

		python compute_scores_pengxu_2sets_20240609.py \
			--task "longbook_choice_eng" \
			--pxout_ref_json ${pxout_ref_json} \
			--model_name pxlong \
			--use_zero_scrolls
		
		python compute_scores_pengxu_2sets_20240609.py \
			--task "longbook_choice_eng" \
			--pxout_ref_json ${pxout_ref_json} \
			--model_name pxlong
	done
fi

