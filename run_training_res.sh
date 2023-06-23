CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/train.py 	\
  	--module 'res' \
	--model_name "google/flan-t5-base" \
	--num_train_epochs 45 \
	--output_dir "./output/GradRes/3.2e_5.45.22_6"  \
	--train_files  "./data/interim/GradRes/KETOD/train.json" "./data/interim/GradRes/WOW/train.json" "./data/interim/GradRes/ORQuAC/train.json" "./data/interim/GradRes/FUSHEDCHAT/train.json"\
	--val_files   "./data/interim/GradRes/KETOD/val.json" "./data/interim/GradRes/WOW/val.json" "./data/interim/GradRes/ORQuAC/val.json" "./data/interim/GradRes/FUSHEDCHAT/val.json"\
	--batch_size  16 \
	--num_beams   4 \
	--weight_decay  0.3 \
	--learning_rate 2e-5 \
	--gradient_accumulation_steps 8 \
	--with_tracking  \
	--report_to wandb \
	--checkpointing_steps epoch \
	--do_eval_per_epoch 
