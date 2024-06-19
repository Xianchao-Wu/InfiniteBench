#########################################################################
# File Name: 1.eval.yarn.mistral.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 05:20:21 2024
#########################################################################
#!/bin/bash

#python -m ipdb eval_yarn_mistral.py \
#	--task longbook_qa_eng \
#	--model_path "NousResearch/Yarn-Mistral-7b-128k"
# useless now for debug only

#exit 1
# TODO this is for 'longbook_choice_eng'
python eval_yarn_mistral.py \
	--task longbook_choice_eng \
	--model_path "NousResearch/Yarn-Mistral-7b-128k"

# TODO
# use llama3 tokenizer to compute tok.len of prompts
# 2024 06 13

#python -m ipdb eval_yarn_mistral.py --task longbook_choice_eng

#python -m ipdb eval_yarn_mistral.py --task kv_retrieval

