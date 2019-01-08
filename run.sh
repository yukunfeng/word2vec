set -x

array=(penn)
gpu_id="0"
log="./log"
echo "" >> $log
for element in "${array[@]}"
do
    data_path="$HOME/common_corpus/data/50lm/$element"
    save_model="$(basename $data_path).save"
    map_path="$HOME/pytorch_examples/word_lm/data/clustercat/$element.cluster"

    python -u main.py --device "cuda:$gpu_id" --input_freq 5 --note "none" \
    --save $save_model --data $data_path --epoch 5 --emsize 200  >> $log

done
