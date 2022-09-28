# TVLT Pre-Training

After data preparation and before running the training script, please modify data_root command in scripts, e.g.
```
data_root='./dataset'
```

### Howto100m

Download howto100m([link](https://www.di.ens.fr/willow/research/howto100m/)) meta files and videos and organize the data structures as below

```
Dataset
│
├── pretrain_dataset                   
│   ├── caption.json
│   └── videos_ht
```

### Yttemporal


Download yttemporal([link](https://rowanzellers.com/merlot/#data)) meta files and videos and organize the data structures as below

```
Dataset
│
├── pretrain_dataset      
│   └── videos_yt

```


# Pretraining
Pretraining Script
```
bash scripts/pretrain_mae_vam.sh
```
