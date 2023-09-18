export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=0,1 WANDB_API_KEY=74956cec51ca7a4366d903f2f3aecba161707d82 accelerate launch
        --config_file src/config/config_fsdp_t5.yaml src/models/train.py \
        --module 'dst' \
        --model_name "google/flan-t5-base" \
        --max_target_length 400 \
        --num_train_epochs 1 \
        --output_dir "/kaggle/working/GradSearch1809/"  \
        --train_files  "/kaggle/input/gradsearch-v2-pro/GradSearch_v2/SGD/train.json" "/kaggle/input/gradsearch-v2-pro/GradSearch_v2/FUSEDCHAT/train.json" \
        --val_files   "/kaggle/input/gradsearch-v2-pro/GradSearch_v2/SGD/val.json" "/kaggle/input/gradsearch-v2-pro/GradSearch_v2/FUSEDCHAT/val.json" \
        --batch_size  8 \
        --num_beams   4 \
        --weight_decay  0.3 \
        --learning_rate 2e-5 \
        --gradient_accumulation_steps 16 \
        --with_tracking  \
        --report_to wandb \
        --checkpointing_steps epoch \
        --do_eval_per_epoch \
        --max_train_samples 500
