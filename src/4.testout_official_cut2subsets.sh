#########################################################################
# File Name: 4.test_cut2subsets.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Jun 29 13:05:24 2024
#########################################################################
#!/bin/bash

# split test.json by info included in 3.prompts_cut2subsets.sh.log

datetime=$(date +%Y%m%d-%H%M%S)
maindir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/results/"
logfn="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/src/3.prompts_cut2subsets.sh.log"

for data_name in "longbook_qa" "longbook_choice"
do
	task_name="${data_name}_eng"

	tfn1="$maindir/gpt4/preds_${task_name}.jsonl" # test out file TODO

	for tstfn in $tfn1
	do
		tstfn_same="$tstfn.same.json"
		tstfn_cut="$tstfn.cut.json"
		if [ ! -e ${tstfn_same} ]
		then
			python cut_testout_json_official.py $logfn $tstfn ${data_name}
		fi

		for aflag in "same" "cut" "all"
		do
			pxout_ref_json="${tstfn}.${aflag}.json"
			if [ $aflag == "all" ]; then
				pxout_ref_json=${tstfn}
			fi

			if [ ${task_name} == "longbook_choice_eng" ]; then
				python compute_scores_pengxu_2sets_20240609.py \
					--task ${task_name} \
					--pxout_ref_json ${pxout_ref_json} \
					--model_name pxlong \
					--use_zero_scrolls
			fi

			# step 2: compute accuracy scores
			python compute_scores_pengxu_2sets_20240609.py \
				--task ${task_name} \
				--pxout_ref_json ${pxout_ref_json} \
				--model_name pxlong
		done
	done
done

