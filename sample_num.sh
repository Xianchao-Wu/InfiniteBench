#########################################################################
# File Name: sample_num.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Jun 26 10:52:27 2024
#########################################################################
#!/bin/bash

for afile in `ls *.txt`
do
        echo $afile

        grep "assistant: " $afile | wc -l
        grep "assistant: NA" $afile | wc -l

        echo "----"
done

