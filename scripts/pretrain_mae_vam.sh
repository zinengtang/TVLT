python run.py with data_root='./dataset' gpus=[0,1,2,3] num_nodes=1 task_mae_vam \
per_gpu_batchsize=4 val_check_interval=0.1 num_workers=12 warmup_steps=1000 \
load_local_path='mae_pretrain_vit_base_full.pth' strict_load=False
