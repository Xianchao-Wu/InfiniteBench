#########################################################################
# File Name: 2.compute.scores.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 07:52:01 2024
#########################################################################
#!/bin/bash

# (A) is also correct TODO
# 0.7205240174672489
python compute_scores_pengxu_2sets_20240609.py \
	--task "longbook_choice_eng" \
	--pxout_ref_json ../../longbook_choice/longbook_choice_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920.2024-06-11-09-24-48.086335.json \
	--model_name pxlong \
	--use_zero_scrolls

# (A) is not correct TODO
# 0.5676855895196506 
python compute_scores_pengxu_2sets_20240609.py \
	--task "longbook_choice_eng" \
	--pxout_ref_json ../../longbook_choice/longbook_choice_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920.2024-06-11-09-24-48.086335.json \
	--model_name pxlong 
