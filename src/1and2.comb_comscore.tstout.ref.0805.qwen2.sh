#########################################################################
# File Name: 1.comb.tstout.ref.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 08:14:18 2024
#########################################################################
#!/bin/bash

#datetime=$(date +%Y%m%d-%H%M%S)
datetime=$(date +%Y%m%d)
echo ${datetime}

#sets7="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240709/long_131072_25_multiturn_qa_blend_continue_pp1/7sets"
#sets7="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240710/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_pp1/7sets/"
#sets7="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240711/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_pp1/7sets"

#sets12="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-12sets"
#sets12="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240715/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_qa_blend_llama3_books_131072_65536_70b_64_3e-7_step_3300_pp1"
#sets12="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-4sets-gpt4turbo/"
sets12="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/prompts-debug-12sets-127k-240804/"

#rootdir="/workspace/asr/megatron.20240606/infbench-20240609/"

refdir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/data"

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

# TODO code_run_5_generate_70b_test_greedy_0_2000_30.txt_0920
# TODO kv_retrieval_5_generate_70b_test_greedy_0_2000_30.txt_0920
# TODO longdialogue_qa_eng_5_generate_70b_test_greedy_0_2000_30.txt_0920
# TODO math_calc_5_generate_70b_test_greedy_0_2000_30.txt_0920
# TODO math_find_5_generate_70b_test_greedy_0_2000_30.txt_0920
# TODO number_string_5_generate_70b_test_greedy_0_2000_30.txt_0920
# TODO passkey_5_generate_70b_test_greedy_0_2000_30.txt_0920

#for afile in `ls $sets12/*/*start*end???`
for afile in `ls $sets12/*/*130048*72b_instruct.txt`
do
	echo $afile
	#for task_name in passkey number_string kv_retrieval longbook_sum_eng longbook_qa_eng longbook_qa_chn longbook_choice_eng longdialogue_qa_eng math_calc math_find code_debug code_run 
	#for task_name in code_run kv_retrieval longdialogue_qa_eng math_calc math_find number_string passkey 
	#for task_name in code_run kv_retrieval  
	#for task_name in passkey number_string kv_retrieval longbook_sum_eng longbook_qa_eng longbook_qa_chn longbook_choice_eng longdialogue_qa_eng math_calc math_find code_debug code_run 
	#for task_name in longbook_sum_eng longdialogue_qa_eng
	#for task_name in longbook_sum_eng
	for task_name in longbook_sum_eng longdialogue_qa_eng longbook_qa_eng longbook_choice_eng
	do
		if [[ $task_name == "longbook_sum_eng" || $task_name == "longdialogue_qa_eng" ]];then
			# for longbook_sum_eng and longdialogue_qa_eng TODO
			pxref_json="$refdir/${task_name}.jsonl"
		else
			# for longbook_qa_eng and longbook_choice_eng TODO
			pxref_json="../../${task_name}/test.json"
		fi
		echo $pxref_json $task_name

		if [[ $afile =~ $task_name ]]
		then
			echo "do $task_name for $afile"
			python output_toknum.py $afile
			longbook_eng_eval ${task_name} ${afile} ${pxref_json}

			afile2=$afile.keep1k
			python output_cut.py $afile $afile2
			#pxref_json="../../${task_name}/test.json"
			#pxref_json="$refdir/${task_name}.jsonl"
			longbook_eng_eval ${task_name} ${afile2} ${pxref_json}
			echo "--------"
			#break
		fi
		#break
	done
	#break
	echo "final display: "
done

