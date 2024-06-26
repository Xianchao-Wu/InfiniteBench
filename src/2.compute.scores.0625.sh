#########################################################################
# File Name: 2.compute.scores.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 07:52:01 2024
#########################################################################
#!/bin/bash

# (A) is also correct TODO
# 0.7205240174672489
adir="/workspace/asr/megatron.20240606/infbench-20240609/longbook_choice_0625"
fn1="$adir/full.txt.2024-06-25-03-58-24.453671.json"
fn2="$adir/ret.txt.2024-06-25-03-58-31.735001.json"

for afile in $fn1 $fn2
do
	python compute_scores_pengxu_2sets_20240609.py \
		--task "longbook_choice_eng" \
		--pxout_ref_json $afile \
		--model_name pxlong \
		--use_zero_scrolls

	# (A) is not correct TODO
	# 0.5676855895196506 
	python compute_scores_pengxu_2sets_20240609.py \
		--task "longbook_choice_eng" \
		--pxout_ref_json $afile \
		--model_name pxlong 
done


adir="/workspace/asr/megatron.20240606/infbench-20240609/longbook_qa_0625"
fn1="$adir/full.txt.2024-06-25-03-58-11.547569.json"
fn2="$adir/ret.txt.2024-06-25-03-58-13.700537.json"

for afile in $fn1 $fn2
do
	python compute_scores_pengxu_2sets_20240609.py \
		--task "longbook_qa_eng" \
		--pxout_ref_json $afile \
		--model_name pxlong
done
