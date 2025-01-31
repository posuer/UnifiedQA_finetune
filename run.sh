python fine-tune.py \
    --model_size base \
    --model_path models/unifiedqa_trained/base/model.ckpt-1100400.index \
    --task_name socialiqa \
    --output_dir output/socialiqa/ \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --save_steps 200 \
    --max_seq_length 128 \
    --max_target_length 10 \
    --do_train \
    --do_eval \
    --do_test 
# --only_save_best_ckpt
