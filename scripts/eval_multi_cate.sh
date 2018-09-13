#! /bin/bash

# Fewshot pretraining and Multi-category
for data in "few_shot_train_pose0.01" "few_shot_train_pose0.1" "few_shot_train_pose0.5" "few_shot_train_pose1.0"; do
for setting in "test" "val"; do
for ckpt_name_meta in `ls log/train_gan_cyc_encdec_randomvp/${data}/*/model-best-*-*.ckpt.meta`; do
    ckpt_basename=`basename $ckpt_name_meta | cut -d'.' -f 1`
    ckpt_dirname=`dirname $ckpt_name_meta`
    ckpt_name="${ckpt_dirname}/${ckpt_basename}.ckpt"
    python inference.py \
        --data ${data}\
        --setting ${setting} \
        --resume ${ckpt_name}
done
done
python scripts/get_score.py --prefix $data # --categories None since it evaluates on all catgories
done


