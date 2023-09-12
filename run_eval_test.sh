CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /kaggle/working/gradient_server_test/src/config/config_fsdp.yaml /kaggle/working/gradient_server_test/src/models/test_evaluation.py 	\
	--module 'dst_tod' \
	--test_files '/kaggle/input/interim-data/GradSearch/GradSearch/SGD/test.json' \
  --batch_size 8 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/final-checkpoint-gradtod/model_dst_3.bin'\
	--log_input_label_predict '/kaggle/working/SGD.json'
