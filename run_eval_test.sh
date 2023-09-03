CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'dst_tod' \
	--test_files '/kaggle/input/mw-extra/MW22/test_tod.json' \
  --batch_size 20 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/final-checkpoint-gradtod/model_dst.bin'\
	--log_input_label_predict '/kaggle/working/MW22.json'
