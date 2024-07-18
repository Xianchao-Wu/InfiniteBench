#########################################################################
# File Name: 1.comb.tstout.ref.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 11 08:14:18 2024
#########################################################################
#!/bin/bash

#datetime=$(date +%Y%m%d-%H%M%S)
datetime=$(date +%Y%m%d)
echo ${datetime}

rootdir="/workspace/asr/megatron.20240606/infbench-20240609/"

logfn="${rootdir}/InfiniteBench/src/3.prompts_cut2subsets.sh.log"

#indir="$rootdir/InfiniteBench/prompts-debug-ret5"
#indir="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240711/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_pp1/7sets"

indir="/workspace/asr/megatron.20240606/infbench-20240609/eval_0708/to_xianchao_20240711/long_131072_25_multiturn_qa_blend_commercial_v28_9_multiturn_pp1/7sets"

# TODO works for both "longbook_qa_eng" and "longbook_choice_eng"
function longbook_eng_eval(){
	data_name=$1 # TODO "longbook_qa_eng" and "longbook_choice_eng"
	pxout_txt=$2
	pxref_json=$3	

	task_name="${data_name}_eng"

	#pxref_json="../../${data_name}/test.json"
	pxout_ref_json="${pxout_txt}.${datetime}.json"

	python prepare_json_from_pxout.py --task ${task_name} \
		--pxout_txt ${pxout_txt} \
		--pxref_json ${pxref_json} \
		--pxout_ref_json ${pxout_ref_json} \
		--sep_by_assistant

	if [[ $task_name =~ "longbook_choice" ]]
	then
		python compute_scores_pengxu_2sets_20240609.py \
			--task ${task_name} \
			--pxout_ref_json ${pxout_ref_json} \
			--model_name pxlong \
			--use_zero_scrolls
	fi

	python compute_scores_pengxu_2sets_20240609.py \
		--task ${task_name} \
		--pxout_ref_json ${pxout_ref_json} \
		--model_name pxlong
}

# TODO 这是针对longbook_qa和longbook_choice的，这两个数据集下，考虑full, same和cut
# full = 使用所有prompts
# cut = 按照128k，进行了剪裁之后的；
# same = 本来的prompt的长度就没有超过128k的!

for afile in `ls $indir/*.txt_0920`
do
	echo $afile
	for data_name in "longbook_qa" "longbook_choice"
	do
		task_name="${data_name}_eng"
		if [[ $afile =~ $task_name ]]
		then
			# TODO
			echo "do $task_name for $afile"
			pxref_json="../../${data_name}/test.json"
			longbook_eng_eval ${data_name} ${afile} ${pxref_json}
			echo "--------"

			tstfn_same="$afile.same"
			tstfn_cut="$afile.cut"
			python cut_testout_txt.py $logfn $afile ${data_name}
			if [ ! -e ${tstfn_same} ]
			then
				python cut_testout_txt.py $logfn $afile ${data_name}
			fi

			for aflag in "same" "cut"
			do
				pxout_txt=${afile}.${aflag}
				pxref_json="${rootdir}/${data_name}/test_${aflag}.json"
				longbook_eng_eval ${data_name} ${pxout_txt} ${pxref_json}
			done
			echo "========"
		fi
		#break
	done
	#break
	echo "final display: "
done

