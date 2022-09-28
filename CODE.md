# Code Structures
```
TVLT
│
├── assets                          # Illustrations                     
│   ├── teaser.png       
│   └── ...
│ 
├── demo_samples                    # Demo samples
│   ├── MOSEI                       
│   │   ├── K0m1tO3Ybyc_2.mp4        
│   │   └── ...
│   ├── Howto100m                   
│   │   ├── mg7Q4o9bNN0.mp4      
│   │   └── ...          
│   └── ...                         # You can add custom data samples
│
├── model                           # Main source
│   ├── data        
│   │   ├── datamodules             # Pytorch-lightning wrap
│   │   │   ├── datamodule_base.py
│   │   │   └── ...          
│   │   └── datasets                # Datasets
│   │   │   ├── vqa_dataset.py     
│   │   │   └── ...    
│   ├── gadgets     
│   │   └── my_metrics.py           # metric utils
│   ├── modules                     
│   │   ├── heads.py                # Model heads
│   │   ├── model_module.py         # pytorch-lightning wrap for model
│   │   ├── model_utils.py          # pytorch-lightning wrap for training metrics
│   │   ├── objectives.py           # pretraining/finetuning objectives
│   │   └── tvlt.py                 # TVLT model
│   └── config.py                   # all configurations
│
├── scripts                         # all scripts
│   ├── finetune_mosei.sh 
│   ├── pretrain_mae_vam.sh
│   └── ... 
│
├── run.py                          # main
├── demo.py                         # utils for demo
├── Demo_Emotional_Analysis.ipynb   # Demo for CMU-MOSEI emotional analysis
├── Demo_Sentiment_Analysis.ipynb   # Demo for CMU-MOSEI sentiment analysis
├── Demo_Video_Audio_MAE.ipynb      # Demo for video/audio reconstruction
└── requirements.txt                
```
