import os
import glob
import json
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def get_mask_from_lengths(lengths, max_len=None, inv=True):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=torch.int32, device=lengths.device)
    if inv:
        mask = ids.unsqueeze(0).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
    else:
        mask = ids.unsqueeze(0).expand(lengths.size(0), -1) < lengths.unsqueeze(1)
    return mask



def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_visual=False)
    mlm_logits = pl_module.transformer.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_mae_video(pl_module, batch, patch_size=16, num_patches=14):
    
    infer = pl_module.infer(batch, mask_text=False, mask_visual=True, use_mae=True)
    pred = pl_module.transformer.mae_score_video(infer["video_feats"])    
    target = patchify_video(infer['video'], patch_size)
    mask = infer['video_masks']
    
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    ret = {
        "mae_video_loss": loss*0.3,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mae_video_loss")(ret["mae_video_loss"])

    pl_module.log(f"mae_video/{phase}/loss", loss)

    return ret


def patchify_video(vids, p=16):
    """
    imgs: (N, T, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    t = vids.shape[1]
    h = vids.shape[3] // p
    w = vids.shape[4] // p
    x = vids.reshape(shape=(vids.shape[0], t, vids.shape[2], h, p, w, p))
    x = torch.einsum('ntchpwq->nthwpqc', x)
    x = x.reshape(shape=(vids.shape[0], h * w * t, p**2 * vids.shape[2]))
    return x
    
    
def compute_mae_audio(pl_module, batch, audio_patch_size=[2,128]):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=True, use_mae=True)
    
    pred = pl_module.transformer.mae_score_audio(infer["audio_feats"])   
    target = patchify_audio(infer['audio'], audio_patch_size[0], audio_patch_size[1])
    mask = infer['audio_masks']
    
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {
        "mae_audio_loss": loss*3.0,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mae_audio_loss")(ret["mae_audio_loss"])

    pl_module.log(f"mae_audio/{phase}/loss", loss)

    return ret


def patchify_audio(audios, p1=16, p2=16):
    """
    audios: (N, 1, H, W)
    x: (N, L, patch_size**2 *3)
    """
    h = audios.shape[2] // p1
    w = audios.shape[3] // p2
    x = audios.reshape(shape=(audios.shape[0], audios.shape[1], h, p1, w, p2))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(audios.shape[0], h * w, p1 * p2 * audios.shape[1]))
    return x
    
    
def denormalize(x):
    return (np.clip(x, -1.0, 1.0) + 1.0)/2.0


def compute_mae_joint(pl_module, batch, patch_size=16, audio_patch_size=[2,128]):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=True, use_mae=True)
    
    pred_a = pl_module.transformer.mae_score_audio(infer["audio_feats"])   
    target_a = patchify_audio(infer['audio'], audio_patch_size[0], audio_patch_size[1])
    mask_a = infer['audio_masks']
    loss_a = (pred_a - target_a) ** 2
    loss_a = loss_a.mean(dim=-1)  # [N, L], mean loss per patch
    loss_a = (loss_a * mask_a).sum() / mask_a.sum()   # mean loss on removed patches

    pred_v = pl_module.transformer.mae_score_video(infer["video_feats"])    
    target_v = patchify_video(infer['video'], patch_size)
    mask_v = infer['video_masks']    
    loss_v = (pred_v - target_v) ** 2
    loss_v = loss_v.mean(dim=-1)  # [N, L], mean loss per patch
    loss_v = (loss_v * mask_v).sum() / mask_v.sum()   # mean loss on removed patches  

    
    ret = {
        "mae_audio_loss": loss_a,
        "mae_video_loss": loss_v
    }

    phase = "train" if pl_module.training else "val"
    loss_a = getattr(pl_module, f"{phase}_mae_audio_loss")(ret["mae_audio_loss"])
    loss_v = getattr(pl_module, f"{phase}_mae_video_loss")(ret["mae_video_loss"])

    pl_module.log(f"mae_audio/{phase}/loss_a", loss_a)
    pl_module.log(f"mae_video/{phase}/loss_v", loss_v)

    return ret


def compute_vam(pl_module, batch):
    pos_len = len(batch["audio_data"]) // 2
    neg_len = len(batch["audio_data"]) - pos_len
    vam_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    vam_labels = vam_labels[torch.randperm(vam_labels.size(0))]
    vam_videos = torch.stack(
            [
                ti if vam_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(batch["video_data"], batch["false_video_0"]))
            ]
        )    
    batch = {k: v for k, v in batch.items()}
    batch["video_data"] = vam_videos

    infer = pl_module.infer(batch, mask_text=False, mask_visual=False, use_mae=False)

    vam_logits = pl_module.transformer.matching_score(infer["cls_feats"])
    vam_loss = F.binary_cross_entropy_with_logits(vam_logits.squeeze(), vam_labels.squeeze())
    ret = {
        "vam_loss": vam_loss,
        "vam_logits": vam_logits,
        "vam_labels": vam_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vam_loss")(ret["vam_loss"])
    acc = getattr(pl_module, f"{phase}_vam_accuracy")(
        ret["vam_logits"], ret["vam_labels"]
    )
    pl_module.log(f"vam/{phase}/loss", loss)
    pl_module.log(f"vam/{phase}/accuracy", acc)

    return ret


def compute_vtm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    vtm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    vtm_labels = vtm_labels[torch.randperm(vtm_labels.size(0))]
    vtm_videos = torch.stack(
            [
                ti if vtm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(batch["video_data"], batch["false_video_0"]))
            ]
        )    
    batch = {k: v for k, v in batch.items()}
    batch["video_data"] = vtm_videos

    infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    
    vtm_logits = pl_module.transformer.matching_score(infer["cls_feats"])
    vtm_loss = F.binary_cross_entropy_with_logits(vtm_logits, vtm_labels.unsqueeze(1))
    ret = {
        "vtm_loss": vtm_loss,
        "vtm_logits": vtm_logits,
        "vtm_labels": vtm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vtm_loss")(ret["vtm_loss"])
    acc = getattr(pl_module, f"{phase}_vtm_accuracy")(
        ret["vtm_logits"], ret["vtm_labels"]
    )
    pl_module.log(f"vtm/{phase}/loss", loss)
    pl_module.log(f"vtm/{phase}/accuracy", acc)

    return ret

                
def a2_parse(a):
    if a < 0:
            res = 0
    else: 
            res = 1
    return res

def get_logits_a2(score):
    eyes = torch.eye(2).to(score.device)
    score = torch.tensor([a2_parse(item) for item in score]).to(score.device)
    return eyes[score]


def compute_moseiemo(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    mosei_score = pl_module.transformer.classifier(infer["cls_feats"])
    
    labels = torch.tensor(batch["emolist"]).to(pl_module.device).float()
    mosei_loss = F.binary_cross_entropy_with_logits(mosei_score, labels)
    
    ret = {
        "moseiemo_loss": mosei_loss,
        "moseiemo_score": mosei_score,
        "moseiemo_labels": labels
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_moseiemo_loss")(ret["moseiemo_loss"])
    
    
    happy = getattr(pl_module, f"{phase}_moseiemo_happy")(mosei_score[:, 0:1], labels[:, 0:1])
    sad = getattr(pl_module, f"{phase}_moseiemo_sad")(mosei_score[:, 1:2], labels[:, 1:2])
    angry = getattr(pl_module, f"{phase}_moseiemo_angry")(mosei_score[:, 2:3], labels[:, 2:3])
    fear = getattr(pl_module, f"{phase}_moseiemo_fear")(mosei_score[:, 3:4], labels[:, 3:4])
    disgust = getattr(pl_module, f"{phase}_moseiemo_disgust")(mosei_score[:, 4:5], labels[:, 4:5])
    surprise = getattr(pl_module, f"{phase}_moseiemo_surprise")(mosei_score[:, 5:6], labels[:, 5:6])
    
    pl_module.log(f"moseiemo/{phase}/loss", loss)
    pl_module.log(f"moseiemo/{phase}/happy", happy)
    pl_module.log(f"moseiemo/{phase}/sad", sad)
    pl_module.log(f"moseiemo/{phase}/angry", angry)
    pl_module.log(f"moseiemo/{phase}/fear", fear)
    pl_module.log(f"moseiemo/{phase}/disgust", disgust)
    pl_module.log(f"moseiemo/{phase}/surprise", surprise)

    return ret


def compute_mosei(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    mosei_score = pl_module.transformer.classifier(infer["cls_feats"])
    
    score_label = torch.tensor(batch["score"]).to(pl_module.device)
    mosei_loss = F.mse_loss(mosei_score.squeeze(), score_label.squeeze(), size_average=None, reduce=None, reduction='mean') 
    ret = {
        "mosei_loss": mosei_loss,
        "mosei_score": mosei_score,
        "mosei_labels2": torch.tensor(batch["label2"]).to(pl_module.device),
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mosei_loss")(ret["mosei_loss"])
    
    acc2 = getattr(pl_module, f"{phase}_mosei_accuracy2")(
        get_logits_a2(mosei_score), ret["mosei_labels2"]
    )
    
    pl_module.log(f"mosei/{phase}/loss", loss)
    pl_module.log(f"mosei/{phase}/accuracy2", acc2)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    vqa_logits = pl_module.transformer.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

  
def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output

def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")

    
def compute_vam(pl_module, batch):
    pos_len = len(batch["audio_data"]) // 2
    neg_len = len(batch["audio_data"]) - pos_len
    vam_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    vam_labels = vam_labels[torch.randperm(vam_labels.size(0))]
    vam_videos = torch.stack(
            [
                ti if vam_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(batch["video_data"], batch["false_video_0"]))
            ]
        )    
    batch = {k: v for k, v in batch.items()}
    batch["video_data"] = vam_videos

    infer = pl_module.infer(batch, mask_text=False, mask_visual=False, use_mae=False)

    vam_logits = pl_module.transformer.matching_score(infer["cls_feats"])
    vam_loss = F.binary_cross_entropy_with_logits(vam_logits.squeeze(), vam_labels.squeeze())
    ret = {
        "vam_loss": vam_loss,
        "vam_logits": vam_logits,
        "vam_labels": vam_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vam_loss")(ret["vam_loss"])
    acc = getattr(pl_module, f"{phase}_vam_accuracy")(
        ret["vam_logits"], ret["vam_labels"]
    )
    pl_module.log(f"vam/{phase}/loss", loss)
    pl_module.log(f"vam/{phase}/accuracy", acc)

    return ret

    

@torch.no_grad()
def compute_vrar_recall(pl_module):
    audio_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(audio_only=True)
    audio_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    audio_loader = torch.utils.data.DataLoader(
        audio_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=False,
        collate_fn=functools.partial(
            audio_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    video_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        video_only=True
    )
    video_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(video_dset, shuffle=False)
    video_loader = torch.utils.data.DataLoader(
        video_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=False,
        sampler=dist_sampler,
        collate_fn=functools.partial(
            video_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    audio_preload = list()
    for _b in tqdm.tqdm(audio_loader, desc="audio prefetch loop"):
        audio_preload.append(
            [_b["audio_data"].to(pl_module.device),
             _b["a_index"],
            ],
        )
        
    aids = list()
    for pre in audio_preload:
        aids += pre[1]
    aids = torch.tensor(aids)

    video_preload = list()
    for _b in tqdm.tqdm(video_loader, desc="video prefetch loop"):
        video_preload.append(
            [_b["video_data"].to(pl_module.device),
             _b["v_index"],
            ],
            
        )
        
    rank_scores = list()
    rank_vids = list()

    for video_batch in tqdm.tqdm(video_preload, desc="rank loop"):

        _ve, _vid = video_batch
        _, l, c, h, w = _ve.shape

        video_batch_score = list()
        for audio_batch in audio_preload:
            fblen = len(audio_batch[0])
            ve = _ve.expand(fblen, l, c, h, w)
            
            with torch.cuda.amp.autocast():
                score = pl_module.transformer.matching_score(
                    pl_module.infer(
                        {
                        "audio_data": audio_batch[0],
                        "video_data": ve,
                        }
                    )["cls_feats"]
                )
                score = F.sigmoid(score)
            video_batch_score.append(score)

        video_batch_score = torch.cat(video_batch_score)
        rank_scores.append(video_batch_score.cpu().tolist()) 
    torch.distributed.barrier()
    
    
    av_rank_scores = np.array(rank_scores).transpose(1,0,2)
    print(av_rank_scores.shape)
    av_rank = {}
    for i, item in enumerate(av_rank_scores):
        av_rank[i] = np.concatenate([np.arange(len(item))[:, None], item], 1)
    
    vr_r1 = torch.tensor(0.0)
    vr_r5 = torch.tensor(0.0)
    vr_r10 = torch.tensor(0.0)
    ar_r1 = torch.tensor(0.0)
    ar_r5 = torch.tensor(0.0)
    ar_r10 = torch.tensor(0.0)
    
    for i_a in av_rank:
        tmp = sorted(av_rank[i_a], key=lambda d: -d[1])
        p = [d[0] for d in tmp].index(i_a)+1

        if p<=1:
            vr_r1 += 1.0/len(av_rank)
        if p<=5:
            vr_r5 += 1.0/len(av_rank)
        if p<=10:
            vr_r10 += 1.0/len(av_rank)
            
    va_rank_scores = np.array(rank_scores)
    va_rank = {}
    for i, item in enumerate(rank_scores):
        va_rank[i] = np.concatenate([np.arange(len(item))[:, None], item], 1)
    
    for i_v in va_rank:
        tmp = sorted(va_rank[i_v], key=lambda d: -d[1])
        p = [d[0] for d in tmp].index(i_v)+1

        if p<=1:
            ar_r1 += 1.0/len(va_rank)
        if p<=5:
            ar_r5 += 1.0/len(va_rank)
        if p<=10:
            ar_r10 += 1.0/len(va_rank)

    return (vr_r1, vr_r5, vr_r10, ar_r1, ar_r5, ar_r10)

    
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

