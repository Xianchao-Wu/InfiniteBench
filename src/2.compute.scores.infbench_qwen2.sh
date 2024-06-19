#########################################################################
# File Name: 2.compute.scores.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 07:52:01 2024
#########################################################################
#!/bin/bash

outdir="/workspace/asr/megatron.20240606/qwen/to.openai.infbench"

# (A) is also correct TODO
python compute_scores_pengxu_2sets_20240609.py \
	--task "longbook_choice_eng" \
	--pxout_ref_json $outdir/prompts_longbook_choice_eng_131022.json.qwen2_72b_instruct.txt.2024-06-19-05-07-26.862837.json \
	--model_name pxlong \
	--use_zero_scrolls

# choice
# (A) is not correct TODO
python compute_scores_pengxu_2sets_20240609.py \
	--task "longbook_choice_eng" \
	--pxout_ref_json $outdir/prompts_longbook_choice_eng_131022.json.qwen2_72b_instruct.txt.2024-06-19-05-07-26.862837.json \
	--model_name pxlong 

# qa

python compute_scores_pengxu_2sets_20240609.py \
	--task "longbook_qa_eng" \
	--pxout_ref_json $outdir/prompts_longbook_qa_eng_131022.json.qwen2_72b_instruct.txt.2024-06-19-05-02-41.768851.json \
	--model_name pxlong
