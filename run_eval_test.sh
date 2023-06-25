CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'res' \
	--test_files '/kaggle/input/interim-data/interim/GradRes/WOW/test.json' \
  --batch_size 20 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/epoch-32-res/epoch_32/pytorch_model.bin'\
	--log_input_label_predict '/kaggle/working/wow.json'
