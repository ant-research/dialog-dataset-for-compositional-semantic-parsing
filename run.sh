#!/bin/bash
set -euox pipefail

dataset=${1-simple}    # default: simple
cuda=${2-0}            # default: 0
rand_seed=${3-1}       # default: 1

# modifiy these locations
output_root=out
input_root=./
wordvector=./cc.zh.300.vec

output=${output_root}/$dataset.r${rand_seed}/output
train_log=${output}/train.log

# clean old output
rm -rf ${output}

input_data="$input_root/$dataset"
onmt_temp_dir="${output}/temp"
test_out="${output}/test_out"
mkdir -p "${onmt_temp_dir}"

# create OpenNMT binarized data
src_tok_max_ntokens=150
tgt_max_ntokens=200
train_num_datapoints=100000
#train_num_datapoints=$(wc -l ${input_data}/train.jsonl)
nbest=1

#convert data to src + tgt, and tokenize
convert_data="$input_root/util.py --convert_data"
python $convert_data --infile $input_data/train.jsonl --outsrc $onmt_temp_dir/train.src_tok --outtgt $onmt_temp_dir/train.tgt_tok
python $convert_data --infile $input_data/dev.jsonl --outsrc $onmt_temp_dir/valid.src_tok --outtgt $onmt_temp_dir/valid.tgt_tok
python $convert_data --infile $input_data/test.jsonl --outsrc $onmt_temp_dir/test.src_tok --outtgt $onmt_temp_dir/test.tgt_tok

dropout=0.7
layers=2
rnn_size=512
train_batch_size=64
test_batch_size=64
valid_batch_size=64

onmt_preprocess \
    --dynamic_dict \
    --train_src ${onmt_temp_dir}/train.src_tok \
    --train_tgt ${onmt_temp_dir}/train.tgt_tok \
    --valid_src ${onmt_temp_dir}/valid.src_tok \
    --valid_tgt ${onmt_temp_dir}/valid.tgt_tok \
    --src_seq_length ${src_tok_max_ntokens} \
    --tgt_seq_length ${tgt_max_ntokens} \
    --src_words_min_frequency 0 \
    --tgt_words_min_frequency 0 \
    --save_data ${onmt_temp_dir}/data


# embedding
onmt_embeddings_dir="${output}/onmt_embeddings"
mkdir -p "${onmt_embeddings_dir}"
python -m embeddings_to_torch \
    -emb_file_both ${wordvector} \
    -dict_file ${onmt_temp_dir}/data.vocab.pt \
    -output_file ${onmt_embeddings_dir}/embeddings

## train OpenNMT models
onmt_models_dir="${output}/onmt_models"
mkdir -p "${onmt_models_dir}"

# approximately validate at each epoch
valid_steps=$(python3 -c "from math import ceil; print(ceil(${train_num_datapoints}/${train_batch_size}))")

CUDA_VISIBLE_DEVICES=$cuda onmt_train \
    --encoder_type brnn \
    --decoder_type rnn \
    --rnn_type LSTM \
    --global_attention general \
    --global_attention_function softmax \
    --generator_function softmax \
    --copy_attn_type general \
    --copy_attn \
    --seed $rand_seed \
    --optim adam \
    --learning_rate 0.001 \
    --early_stopping 2 \
    --batch_size ${train_batch_size} \
    --valid_batch_size $valid_batch_size \
    --valid_steps ${valid_steps} \
    --save_checkpoint_steps ${valid_steps} \
    --data ${onmt_temp_dir}/data \
    --pre_word_vecs_enc ${onmt_embeddings_dir}/embeddings.enc.pt \
    --pre_word_vecs_dec ${onmt_embeddings_dir}/embeddings.dec.pt \
    --word_vec_size 300 \
    --attention_dropout 0 \
    --dropout $dropout \
    --layers $layers \
    --rnn_size $rnn_size \
    --gpu_ranks 0 \
    --world_size 1 \
    --keep_checkpoint 4 \
    --save_model ${onmt_models_dir}/checkpoint  > $train_log 2>&1

# get th Best
checkpoint=`grep Best $train_log | perl -nle 'if (/at step (\d+)/) { print $1;}'`
onmt_model_pt="${onmt_models_dir}/checkpoint_step_${checkpoint}.pt"

mkdir -p "${test_out}"

# predict programs on the test set using a trained OpenNMT model
CUDA_VISIBLE_DEVICES=$cuda onmt_translate \
    --model ${onmt_model_pt} \
    --max_length ${tgt_max_ntokens} \
    --src ${onmt_temp_dir}/test.src_tok \
    --replace_unk \
    --n_best ${nbest} \
    --batch_size $test_batch_size \
    --beam_size 10 \
    --gpu 0 \
    --report_time \
    --output ${test_out}/test.nbest
# convert pred to json
convert_data="$input_root/util.py --convert_pred"
test_pred_json="${test_out}/test.nbest.json"
python $convert_data --infile ${test_out}/test.nbest --outfile $test_pred_json

# scoring
python scoring/scorer.py $input_data/test.jsonl $test_pred_json -N eval_norm.json



