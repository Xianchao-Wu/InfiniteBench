#########################################################################
# File Name: 1.comb.tstout.ref.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 08:14:18 2024
#########################################################################
#!/bin/bash

# TODO good for 'qa' dataset
test1=1
if [ $test1 -eq 1 ]
then
	task_name="longbook_qa_eng"
	fn1="../../longbook_qa_0625/full.txt"
	fn2="../../longbook_qa_0625/ret.txt"
	for pxout_txt in $fn1 $fn2 
	do
		#pxout_txt="../../longbook_qa/longbook_qa_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920"
		pxref_json="../../longbook_qa/test.json"
		python prepare_json_from_pxout.py --task ${task_name} \
			--pxout_txt ${pxout_txt} \
			--pxref_json ${pxref_json}
	done
fi

# TODO good for 'choice' dataset as well
task_name="longbook_choice_eng"
fn1="../../longbook_choice_0625/full.txt"
fn2="../../longbook_choice_0625/ret.txt"
for pxout_txt in $fn1 $fn2
do
	#pxout_txt="../../longbook_choice/longbook_choice_eng.e5_mistral_retriever_chunkbysents1200_5_generate_70b_test_greedy_0_1000_1056_ret.txt_0920"
	pxref_json="../../longbook_choice/test.json"
	python prepare_json_from_pxout.py --task ${task_name} \
		--pxout_txt ${pxout_txt} \
		--pxref_json ${pxref_json}
done
