# warmup
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="0.0.0.0" --master_port=23456 train.py --train True --test False --batch_size 8 --max_len 512 --lr 5e-05 --epochs 3 --model_name_or_path gogamza/kobart-base-v1  --warmup_stage True --model_type bart

# warmup infer
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="0.0.0.0" --master_port=23456 train.py --train False --test True --batch_size 8 --max_len 512 --gpu_ids 0 --model_name_or_path gogamza/kobart-base-v1  --warmup_stage True --model_type bart
