#########################################################################
# File Name: test_eval_infbench_sets12.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Sep 10 00:21:13 2024
#########################################################################
#!/bin/bash

# use sets7 to test:

adir="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240709/long_131072_25_multiturn_qa_blend_continue_pp1/7sets"
refdir="/workspace/asr/megatron.20240606/infbench-20240609/InfiniteBench/data"

#for atask in code_run kv_retrieval longdialogue_qa_eng math_calc math_find number_string passkey
for atask in code_run
do
	echo "--------"
	task_name=$atask
	pxout_txt=$adir/${task_name}_5_generate_70b_test_greedy_0_2000_30.txt_0920
	pxref_json=$refdir/${task_name}.jsonl
	./eval_infbench_sets12.sh ${task_name} ${pxout_txt} ${pxref_json} 
done

adir="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240709/long_131072_25_multiturn_qa_blend_continue_pp1"
for atask in longbook_choice_eng longbook_qa_eng
do
	echo "--------"
	task_name=$atask
	pxout_txt=$adir/${task_name}.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_10.txt_0920
	pxref_json=$refdir/${task_name}.jsonl
	./eval_infbench_sets12.sh ${task_name} ${pxout_txt} ${pxref_json} 
done

