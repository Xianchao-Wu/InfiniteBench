#########################################################################
# File Name: 1.eval.yarn.mistral.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 05:20:21 2024
#########################################################################
#!/bin/bash

# TODO this is for 'longbook_qa_eng' and 'longbook_choice_eng'

#for atask in longbook_qa_eng longbook_choice_eng
#for atask in longbook_qa_eng
#for atask in longbook_qa_eng longbook_choice_eng 

maxlen=126976
for atask in longbook_qa_eng longbook_choice_eng longbook_sum_eng longdialogue_qa_eng longbook_qa_chn code_debug code_run math_calc math_find passkey number_string kv_retrieval   
do
{
	echo $atask
	#for mname in "claude2" "gpt4" "kimi" "yarn-mistral"
	#for mname in "kimi" "yarn-mistral"
	for mname in "claude2" "gpt4" "kimi" "yarn-mistral"
	do
	{
		python eval_obtain_prompts.py \
			--task $atask \
			--model_path "Qwen/Qwen2-72B-Instruct" \
			--cache_dir "/workspace/asr/megatron.20240606/qwen" \
			--output_dir "../prompts-debug-12sets/" \
			--model_name $mname \
			--max_seq_len $maxlen > eval_obtain_prompts.py.log.$atask.$mname.$maxlen.20240810 2>&1
	} &
	done
	wait
} &
done
wait

echo "done all 12 sets."
