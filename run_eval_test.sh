CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'res' \
	--test_files '/kaggle/input/interim-data/interim/GradRes/FUSHEDCHAT/test.json' \
  --batch_size 20 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir '/kaggle/input/epoch44-res-30-6/pytorch_model_epoch44_res_30-06.bin'\
	--log_input_label_predict '/kaggle/working/fc.json'
