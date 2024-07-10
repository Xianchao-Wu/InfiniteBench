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
	ret5_dir="/workspace/asr/megatron.20240606/rag_data_0708/"
	ret5_file="$ret5_dir/${atask}.e5_mistral_retriever_chunkbysents1200/test.json"
	for mname in "gpt4" 
	do
		#for topn in 5 10 20 30 40 50
		for topn in 60 70 80 90 100
		do
			python eval_obtain_prompts_ret5.py \
				--task $atask \
				--ret5_file $ret5_file \
				--model_path "Qwen/Qwen2-72B-Instruct" \
				--cache_dir "/workspace/asr/megatron.20240606/qwen" \
				--output_dir "../prompts-debug-ret5/" \
				--model_name $mname \
				--topn $topn
		done
	done
done
