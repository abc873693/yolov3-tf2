
#declare -a experitentDir=("20200326_1" "20200326_2" "20200326_3" "20200408_1" "20200408_2" "20200408_3" "20200409_2" "20200410_1" "20200410_2" "20200410_3" "20200410_4")
declare -a experitentDirs=("20200423_6")
#declare -a dataset=("brown128" "green859" "thickBrown1227" "thinGreen1068")
declare -a dataset=("dataset/microfield_monocular_size_v2/416X416/cfg/valid.txt" "dataset/microfield_monocular_size_v2/416X416/cfg/test.txt")

weightName=best

#yolov3-tiny single-output
model=yolov3-tiny 

size=416

rm records.txt

for experitentDir in "${experitentDirs[@]}"
do
    echo "--------------$experitentDir--------------"
    # mkdir -p ex/results/$experitentDir/dataset/
    for data in "${dataset[@]}"
    do
        echo "--------------$data--------------"
        CUDA_VISIBLE_DEVICES=6 python3 detect_dataset.py --weights_postfix $weightName --classes ./data/shrimp.names --model $model --num_classes 1 --dataset $data --experiment $experitentDir --map
    done
done
