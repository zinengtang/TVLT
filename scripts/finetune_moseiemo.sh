python run.py with data_root='./dataset/CMU_MOSEI/' gpus=[0,1,2,3] num_nodes=1 task_cls_moseiemo \
per_gpu_batchsize=4 num_workers=16 val_check_interval=0.2 warmup_steps=100 max_epoch=10 \
load_hub_path='TVLT.ckpt'
# load_local_path='.'
