from sacred import Experiment

ex = Experiment("TVLT")

def _loss_names(d):
    ret = {
        "vam": 0,
        "vtm": 0,
        "mae_audio": 0,
        "mae_video": 0,
        "vqa": 0,
        "mlm": 0,
        "mosei": 0,
        "moseiemo": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "TVLT"
    seed = 0
    datasets = []
    loss_names = _loss_names({})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    max_text_len = 40
    draw_false_text = 0
    tokenizer = "bert-base-uncased" # tokenizer for text
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    use_text = False
    
    # Video setting
    video_size = 224 # video frame reshape size
    draw_false_video = 0 # draw negative video for video-audio matching
    video_only = False
    max_frames = 64 # max frames of frame position embedding
    num_frames = 8 # number frames to use for input video
    use_video = True
    
    # Audio Setting
    audio_size = 1024 # max audio spectrogram
    frequency_size = 128 # frequency axis size
    max_audio_patches = 1020 # max temporal position embedding
    draw_false_audio = 0 # draw negative audio
    use_audio = True
    audio_only = False
    frame_masking = False # frame level audio masking

    # Transformer Setting
    model_type = "mae_vit_base_patch16_dec512d8b" # model configuration
    patch_size = 16
    audio_patch_size = [16, 16]
    hidden_size = 768
    decoder_hidden_size = 512
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    use_mae = False
    drop_rate = 0.1
    
    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.001
    decay_power = 1
    max_epoch = 100
    max_steps = 10
    warmup_steps = 2500

    # Downstream Setting
    vqav2_label_size = 3129 
    get_va_recall_metric = False # perform audio video retrieval at end of each epoch

    # PL Trainer Setting
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    strict_load = False
    load_local_path = ""
    load_hub_path = ""
    num_workers = 32

    
@ex.named_config
def task_cls_moseiemo():
    exp_name = "cls_moseiemo"
    datasets = ["moseiemo"]
    loss_names = _loss_names({"moseiemo": 1})
    model_type='mae_vit_base_patch16_dec512d8b'
    batch_size = 128
    audio_size = 1024
    num_frames = 8
    use_video = True
    use_audio = True 
    use_text = False
    learning_rate = 1e-5
    max_epoch = 10
    

@ex.named_config
def task_cls_mosei():
    exp_name = "cls_mosei"
    datasets = ["mosei"]
    loss_names = _loss_names({"mosei": 1})
    model_type='mae_vit_base_patch16_dec512d8b'
    batch_size = 128
    audio_size = 1024
    num_frames = 8
    use_video = True
    use_audio = True 
    use_text = False
    learning_rate = 1e-5
    max_epoch = 10
       

@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    model_type='mae_vit_base_patch16_dec512d8b'
    batch_size = 128
    audio_size = 1024
    num_frames = 1
    use_video = True
    use_audio = True 
    use_text = False
    learning_rate = 1e-5
    max_epoch = 10


@ex.named_config
def task_finetune_msrvtt():
    exp_name = "finetune_msrvtt"
    datasets = ["msrvtt"]
    loss_names = _loss_names({"vam": 1})
    batch_size = 128
    audio_size = 1024
    num_frames = 8
    use_video = True
    use_audio = True 
    use_text = False
    get_va_recall_metric = True
    draw_false_video = 1
    learning_rate = 1e-5
    max_epoch = 40
    
    
@ex.named_config
def task_mae_vam():
    exp_name = "mae_vam"
    datasets = ["howto100m", "yttemporal"]
    loss_names = _loss_names({"vam": 1, "mae_video": 1, "mae_audio": 1})
    batch_size = 4096
    audio_size = 1024
    num_frames = 4
    use_video = True
    use_audio = True 
    use_text = False
    draw_false_video = 1 
    use_mae = True
    learning_rate = 1e-5
    max_epoch = 100
    
   
