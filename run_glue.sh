export CUDA_VISIBLE_DEVICES=2
export GLUE_DIR=./data/sst/fine-tune
export TASK_NAME=SST-2

python ./run_glue.py \
    --model_type bert \
    --model_name_or_path /home/zyzeng/pretrained/transformers/bert-base-uncase \
    --tokenizer_name /home/zyzeng/pretrained/transformers/bert-base-uncase \
    --config_name /home/zyzeng/pretrained/transformers/bert-base-uncase \
    --task_name $TASK_NAME \
    --do_train \
    --do_lower_case \
    --data_dir $GLUE_DIR/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --lamda_ 0.25 \
    --task multi \
    --evaluate_during_training \
    --log_dir log/pretrained/with_tree/lambda2.5 \
    --output_dir /tmp/SST-2/multi-task/lambda2.5