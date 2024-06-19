#########################################################################
# File Name: 1.comb.tstout.ref.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 08:14:18 2024
#########################################################################
#!/bin/bash

# TODO good for 'qa' dataset
test1=0
if [ $test1 -eq 1 ]
then
	task_name="longbook_qa_eng"
	#qwen2out_txt="../../longbook_qa/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920"
	qwen2out_txt="/workspace/asr/megatron.20240606/qwen/to.openai.infbench/prompts_longbook_qa_eng_131022.json.qwen2_72b_instruct.txt"
	pxref_json="../../longbook_qa/test.json"
	python prepare_json_from_pxout.py --task ${task_name} \
		--pxout_txt ${qwen2out_txt} \
		--pxref_json ${pxref_json} \
		--sep_by_assistant
fi

#exit 0

# TODO good for 'choice' dataset as well
task_name="longbook_choice_eng"
#pxout_txt="../../longbook_choice/longbook_choice_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920"
qwen2out_txt="/workspace/asr/megatron.20240606/qwen/to.openai.infbench/prompts_longbook_choice_eng_131022.json.qwen2_72b_instruct.txt"
pxref_json="../../longbook_choice/test.json"
python prepare_json_from_pxout.py --task ${task_name} \
	--pxout_txt ${qwen2out_txt} \
	--pxref_json ${pxref_json} \
	--sep_by_assistant

