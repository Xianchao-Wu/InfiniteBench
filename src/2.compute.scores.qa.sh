#########################################################################
# File Name: 2.compute.scores.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 07:52:01 2024
#########################################################################
#!/bin/bash

#python -m ipdb compute_scores_pengxu_2sets_20240609.py --task "longbook_qa_eng" --pxout_ref_json ../../longbook_qa/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920.2024-06-11-07-41-01.155680.json --model_name pxlong

python compute_scores_pengxu_2sets_20240609.py \
	--task "longbook_qa_eng" \
	--pxout_ref_json ../../longbook_qa/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920.2024-06-11-08-08-59.741131.json \
	--model_name pxlong
