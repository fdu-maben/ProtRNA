python train_ribosome_loading.py \
../data \
--init_params ../weights/ProtRNA.ckpt \
--lm_type ProtRNA \
--test_set human \
--feature_path ../feature \ 
--batch_size 64 -test_only --num_workers 32 --pin_memory --accelerator gpu --max_epochs 50 \
--output_dir ../output

python train_ribosome_loading.py \
../data \
--init_params ../weights/ProtRNA.ckpt \
--lm_type ProtRNA \
--test_set random \
--feature_path ../feature \ 
--batch_size 64 -test_only --num_workers 32 --pin_memory --accelerator gpu --max_epochs 50 \
--output_dir ../output