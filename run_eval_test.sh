CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'dst' \
	--test_files '/kaggle/input/interim-data/interim/GradSearch/KETOD/test.json' \
  --batch_size 20 \
	--num_beams   4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/epoch-18-dst/epoch_18/pytorch_model.bin'
