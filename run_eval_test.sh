CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /kaggle/working/gradient_server_test/src/config/config_fsdp_t5.yaml /kaggle/working/gradient_server_test/src/models/test_evaluation.py 	\
	--module 'dst_tod' \
	--model_name 'google/flan-t5-base' \
	--test_files '/kaggle/input/gradsearch-v2-pro/GradSearch_v2/FUSEDCHAT/test.json' \
  --batch_size 14 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/final-checkpoint-gradtod/pytorch_model_epoch_2.bin'\
	--log_input_label_predict '/kaggle/working/FUSEDCHAT.json'
