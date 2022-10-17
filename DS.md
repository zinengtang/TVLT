# Finetuning TVLT on Downstream Task
After data preparation and before running the training script, please modify data_root command in scripts, e.g.,
```
data_root='./dataset/cmumosei'
```

### CMU-MOSEI

Download CMU-MOSEI [[link]](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) meta files and videos and organize the data structures as below

```
Dataset
│
├── CMU_MOSEI            
│   ├── train
│   │   ├── audio
│   │   ├── video
│   │   └── text
│   ├── valid
│   ├── test
│   ├── labels
│   └── labels_emotion
```

Sentiment analysis
```
bash scripts/finetune_mosei.sh
```

Emotional analysis
```
bash scripts/finetune_moseiemo.sh
```

### VQAv2

Download VQAv2 [[link]](https://visualqa.org/) meta files and audios [[link]](https://nlp.cs.unc.edu/data/TVLT/vqa_dataset/) and images and organize the data structures as below

```
Dataset
│
├── VQA      
│   ├── audios
│   ├── train2014
│   │   ├── 0.jpg
│   │   └── ...
│   ├── val2014
│   ├── test2014
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
```

```
bash scripts/finetune_vqa.sh
```

### MSRVTT

Download meta files from [frozen-in-time](https://github.com/m-bain/frozen-in-time) or directly [[link]](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip) and videos and organize the data structures as below

```
Dataset
│
├── MSRVTT   
│   ├── high-quality
│   ├── structured-symlinks
│   ├── annotation
│   │   └── MSR_VTT.json
│   ├── videos
│   └── raw_audios
```

```
bash scripts/finetune_msrvtt.sh
```
