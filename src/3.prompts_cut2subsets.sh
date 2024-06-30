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
for atask in longbook_qa_eng longbook_choice_eng
do
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
			--output_dir "../prompts-debug-2subsets/" \
			--model_name $mname
	done
done
