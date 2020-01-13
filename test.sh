
#declare -a experitentDir=("20190712" "20190715" "20190716" "20190718" "20190719" "20190720" "20190722-1" "20190722-2" "20190723-1" "20190723-2" "20190723-3")
declare -a experitentDirs=("20191212" "20200104_1" "20200104_2" "20200104_3" "20200108" "20200109_1" "20200109_2" "20200109_3" "20200109_4")
#declare -a dataset=("brown128" "green859" "thickBrown1227" "thinGreen1068")
declare -a dataset=("microfield_5_v2_test" "microfield_6_test")

weightName=last

rm records.txt

for experitentDir in "${experitentDirs[@]}"
do
    echo "--------------$experitentDir--------------"
    # mkdir -p ex/results/$experitentDir/dataset/
    for data in "${dataset[@]}"
    do
        echo "--------------$data--------------"
        python3 detect_dataset.py --weights_postfix $weightName --classes ./data/shrimp.names  --tiny --num_classes 1 --dataset $data --experiment $experitentDir --map
    done
done
