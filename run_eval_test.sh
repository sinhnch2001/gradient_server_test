CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'res' \
	--test_files '/kaggle/input/interim-data/interim/GradRes/FUSHEDCHAT/test.json' \
  --batch_size 26 \
	--num_beams   4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/epoch-13-res/epoch_13/pytorch_model.bin'
