CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file src/config/config_fsdp.yaml src/models/eval_test.py 	\
	--module 'dst_ood' \
	--test_files './data/interim/GradSearch/FUSEDCHAT/test_odd.json' \
  --batch_size 20 \
	--num_beams 4 \
	--with_tracking  \
	--path_to_save_dir 'C:\ALL\OJT\server\gradient_server_test\models\pytorch_model_dst.bin'\
	--log_input_label_predict 'C:\ALL\OJT\server\gradient_server_test\reports\fc_odd.json'
