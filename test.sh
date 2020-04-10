
#declare -a experitentDir=("20190712" "20190715" "20190716" "20190718" "20190719" "20190720" "20190722-1" "20190722-2" "20190723-1" "20190723-2" "20190723-3")
declare -a experitentDirs=("20200410_1" "20200410_2")
#declare -a dataset=("brown128" "green859" "thickBrown1227" "thinGreen1068")
declare -a dataset=("microfield_monocular_test")

weightName=best

rm records.txt

for experitentDir in "${experitentDirs[@]}"
do
    echo "--------------$experitentDir--------------"
    # mkdir -p ex/results/$experitentDir/dataset/
    for data in "${dataset[@]}"
    do
        echo "--------------$data--------------"
        CUDA_VISIBLE_DEVICES=0 python3 detect_dataset.py --weights_postfix $weightName --classes ./data/shrimp.names  --tiny --num_classes 1 --dataset $data --experiment $experitentDir --map
    done
done
