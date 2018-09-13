#! /bin/bash

# Single category evaluation
for cate in "airplanes" "benches" "cars" "chairs" "sofas" "tables"; do
data="${cate}_pose0.5"
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
python scripts/get_score.py --prefix $data --categories $cate
done

