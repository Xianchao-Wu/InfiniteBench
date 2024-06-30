#########################################################################
# File Name: 4.test_cut2subsets.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Jun 29 13:05:24 2024
#########################################################################
#!/bin/bash

# split test.json by info included in 3.prompts_cut2subsets.sh.log

datetime=$(date +%Y%m%d-%H%M%S)

for data_name in "longbook_qa" "longbook_choice"
do
	task_name="${data_name}_eng"
	tstjson="/workspace/asr/megatron.20240606/infbench-20240609/${data_name}/test.json"
	logfn="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/src/3.prompts_cut2subsets.sh.log"
	tst_same_json="/workspace/asr/megatron.20240606/infbench-20240609/${data_name}/test_same.json"
	tst_cut_json="/workspace/asr/megatron.20240606/infbench-20240609/${data_name}/test_cut.json"

	if [ ! -e ${tst_same_json} ]
	then
		# test.json -> test_same.json and test_cut.json
		python cut_test_json.py $logfn $tstjson ${data_name}
	fi
	# TODO qwen2-72b
	# TODO  
	tfn1="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug/${data_name}_eng/prompts_${data_name}_eng_131022_template_gpt4.json.qwen2_72b_instruct.txt"
	tfn2="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug/${data_name}_eng/prompts_${data_name}_eng_131022_template_claude2.json.qwen2_72b_instruct.txt"
	tfn3="/workspace/asr/megatron.20240606/infbench-20240609/${data_name}/${data_name}_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920"
	tfn4="/workspace/asr/megatron.20240606/infbench-20240609/${data_name}_0625/${data_name}_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1100.txt_0920"
	tfn5="/workspace/asr/megatron.20240606/infbench-20240609/${data_name}_0625/${data_name}_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1100_ret.txt_0920"

	tfn6="/workspace/asr/megatron.20240606/infbench-20240609/nqa_128k_long_25_pp1/${data_name}_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_148.txt_0920"
	tfn7="/workspace/asr/megatron.20240606/infbench-20240609/nqa_128k_long_25_pp1/${data_name}_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_148_ret.txt_0920"

	# TODO TODO TODO
	#for tstfn in $tfn1 $tfn2 $tfn3 $tfn4 $tfn5
	for tstfn in $tfn6 $tfn7
	do
		tstfn_same="$tstfn.same"
		tstfn_cut="$tstfn.cut"
		if [ ! -e ${tstfn_same} ]
		then
			python cut_testout_txt.py $logfn $tstfn ${data_name}
		fi

		for aflag in "same" "cut"
		do
			pxout_txt=${tstfn}.${aflag}
			pxout_ref_json="${pxout_txt}.${datetime}.json"
			# TODO compute score here:
			python prepare_json_from_pxout.py --task ${task_name} \
				--pxout_txt ${pxout_txt} \
				--pxref_json "/workspace/asr/megatron.20240606/infbench-20240609/${data_name}/test_${aflag}.json" \
				--pxout_ref_json ${pxout_ref_json} \
				--sep_by_assistant # TODO

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

