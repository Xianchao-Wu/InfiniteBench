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
for atask in longbook_qa_eng longbook_choice_eng longbook_sum_eng longdialogue_qa_eng longbook_qa_chn code_debug code_run math_calc math_find passkey number_string kv_retrieval   
do
{
	echo $atask
	#for mname in "claude2" "gpt4" "kimi" "yarn-mistral"
	#for mname in "kimi" "yarn-mistral"
	#--model_path "Qwen/Qwen2-72B-Instruct" \
	for mname in "claude2" "gpt4" "kimi" "yarn-mistral"
	do
	{
		python eval_obtain_prompts.py \
			--task $atask \
			--model_path "gradientai/Llama-3-70B-Instruct-Gradient-524k" \
			--cache_dir "/workspace/asr/megatron.20240606/llama3" \
			--output_dir "../prompts-debug-12sets/" \
			--model_name $mname \
			--max_seq_len 524238 > eval_obtain_prompts.py.log.$atask.$mname.524k.20240715 2>&1
	} &
	done
	wait
} &
done
wait

echo "done all 12 sets."
