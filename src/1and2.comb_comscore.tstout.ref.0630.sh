#########################################################################
# File Name: 1.comb.tstout.ref.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 08:14:18 2024
#########################################################################
#!/bin/bash

datetime=$(date +%Y%m%d-%H%M%S)
echo ${datetime}

outdir="/workspace/asr/megatron.20240606/infbench-20240609/nqa_128k_long_25_pp1/"

# good for 'qa' dataset
test1=1
if [ $test1 -eq 1 ]
then
	task_name="longbook_qa_eng"
	#fn1="../../longbook_qa_0625/full.txt"
	#fn2="../../longbook_qa_0625/ret.txt"

	fn1="$outdir/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_148.txt_0920"
	fn2="$outdir/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_148_ret.txt_0920"
	for pxout_txt in $fn1 $fn2 
	do
		#pxout_txt="../../longbook_qa/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920"
		pxref_json="../../longbook_qa/test.json"
		pxout_ref_json="${pxout_txt}.${datetime}.json"
		python prepare_json_from_pxout.py --task ${task_name} \
			--pxout_txt ${pxout_txt} \
			--pxref_json ${pxref_json} \
			--pxout_ref_json ${pxout_ref_json} \
			--sep_by_assistant

		python compute_scores_pengxu_2sets_20240609.py \
			--task "longbook_qa_eng" \
			--pxout_ref_json ${pxout_ref_json} \
			--model_name pxlong
	done
fi

# good for 'choice' dataset as well
task_name="longbook_choice_eng"
#fn1="../../longbook_choice_0625/full.txt"
#fn2="../../longbook_choice_0625/ret.txt"
fn1="$outdir/longbook_choice_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_148.txt_0920"
fn2="$outdir/longbook_choice_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_148_ret.txt_0920"
for pxout_txt in $fn1 $fn2
do
	# step 1: combine tst.out and ref:
	pxref_json="../../longbook_choice/test.json"
	pxout_ref_json="${pxout_txt}.${datetime}.json"
	python prepare_json_from_pxout.py --task ${task_name} \
		--pxout_txt ${pxout_txt} \
		--pxref_json ${pxref_json} \
		--pxout_ref_json ${pxout_ref_json} \
		--sep_by_assistant

	python compute_scores_pengxu_2sets_20240609.py \
		--task "longbook_choice_eng" \
		--pxout_ref_json ${pxout_ref_json} \
		--model_name pxlong \
		--use_zero_scrolls

	# step 2: compute accuracy scores
	python compute_scores_pengxu_2sets_20240609.py \
		--task "longbook_choice_eng" \
		--pxout_ref_json ${pxout_ref_json} \
		--model_name pxlong 
done

