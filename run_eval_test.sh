CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'dst_tod' \
	--test_files '/kaggle/input/interim-data/interim_blv1/GradSearch/KETOD/test.json' \
  --batch_size 20 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/epoch44-dst-6-7/module1_pytorch_model.bin'\
	--log_input_label_predict '/kaggle/working/ketod_dst_tod.json'
