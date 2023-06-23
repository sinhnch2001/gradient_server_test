export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/train.py 	\
  	--module 'dst' \
	--model_name "google/flan-t5-base" \
	--num_train_epochs 45 \
	--output_dir "./output/GradSearch-08.06"  \
	--train_files  "./data/interim/GradSearch/KETOD/train.json" "./data/interim/GradSearch/FUSEDCHAT/train.json" \
	--val_files   "./data/interim/GradSearch/KETOD/val.json" "./data/interim/GradSearch/FUSEDCHAT/val.json" \
	--batch_size  16 \
	--num_beams   4 \
	--weight_decay  0.3 \
	--learning_rate 1e-5 \
	--gradient_accumulation_steps 8 \
	--with_tracking  \
	--report_to wandb \
	--checkpointing_steps epoch \
	--do_eval_per_epoch 

