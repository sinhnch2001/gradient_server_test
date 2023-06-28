CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/train.py 	\
  	--module 'res' \
	--model_name "google/flan-t5-base" \
	--num_train_epochs 45 \
	--output_dir "./output/GradRes/25.06.wm1000"  \
	--train_files  "/kaggle/working/gradient_server_test/interim/GradRes/KETOD/train.json" "/kaggle/working/gradient_server_test/interim/GradRes/WOW/train.json" "/kaggle/working/gradient_server_test/interim/GradRes/ORQuAC/train.json" "/kaggle/working/gradient_server_test/interim/GradRes/FUSHEDCHAT/train.json"\
	--val_files   "/kaggle/working/gradient_server_test/interim/GradRes/KETOD/val.json" "/kaggle/working/gradient_server_test/interim/GradRes/WOW/val.json" "/kaggle/working/gradient_server_test/interim/GradRes/ORQuAC/val.json" "/kaggle/working/gradient_server_test/interim/GradRes/FUSHEDCHAT/val.json"\
	--batch_size  16 \
	--num_beams   4 \
	--weight_decay  0.0 \
	--learning_rate 5e-5 \
	--num_warmup_steps 1500 \
	--gradient_accumulation_steps 16 \
	--with_tracking  \
	--report_to wandb \
	--checkpointing_steps epoch \
	--do_eval_per_epoch \
