CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'dst_tod' \
	--test_files '/kaggle/input/mw-extra/MW21/MW21/test.json' \
  --batch_size 18 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/final-checkpoint-gradtod/model_dst.bin'\
	--log_input_label_predict '/kaggle/working/MW21.json'
