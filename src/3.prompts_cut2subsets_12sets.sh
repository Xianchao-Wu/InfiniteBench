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
for atask in longbook_qa_eng longbook_choice_eng longbook_sum_eng longdialogue_qa_eng longbook_qa_chn code_debug code_run math_calc math_find passkey number_string kv_retrieval
do
{
	echo $atask
	#for mname in "claude2" "gpt4" "kimi" "yarn-mistral"
	#for mname in "kimi" "yarn-mistral"
	#for mname in "claude2" "gpt4" "kimi" "yarn-mistral"
	for mname in "gpt4"
	do
		python eval_obtain_prompts_cut2subsets.py \
			--task $atask \
			--model_path "Qwen/Qwen2-72B-Instruct" \
			--cache_dir "/workspace/asr/megatron.20240606/qwen" \
			--output_dir "../prompts-debug-2subsets-12datasets/" \
			--model_name $mname > eval_obtain_prompts_cut2subsets.py.log.$atask.$mname.20240702 2>&1 
	done
} &
done
wait

echo "done. 12 sets"

