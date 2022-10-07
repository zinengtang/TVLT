import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from tqdm import tqdm

from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
from model.modules import heads, objectives

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(
                ~mask[:, None, None, :].bool(), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AudioPatchEmbed(nn.Module):
    """ Audio to Patch Embedding"""

    def __init__(
        self,
        img_size=173,
        patch_size=[16, 16],
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TVLT(nn.Module):

    def __init__(
        self, img_size=224, in_chans=3,
        patch_size=16, audio_patch_size=[2, 128], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm), eps=1e-6,
        config=None,
    ):

        super().__init__()

        self.config = config

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        use_audio = config['use_audio']
        self.use_audio = use_audio
        self.use_mae = config["loss_names"]["mae_audio"] > 0 or config["loss_names"]["mae_video"] > 0
        self.patch_embed_v = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.num_patches_v = self.patch_embed_v.num_patches
        self.frequency_size = config["frequency_size"]
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_size = patch_size
        self.temporal_embed = nn.Parameter(torch.zeros(
            1, config['max_frames'], config["hidden_size"]))
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.num_patches_v, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_audio:
            self.patch_embed_a = AudioPatchEmbed(
                img_size=img_size,
                patch_size=audio_patch_size,
                in_chans=1,
                embed_dim=embed_dim,
            )
            self.audio_patch_size = audio_patch_size
            self.type_embed_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_a = nn.Parameter(torch.zeros(
                1, config['max_audio_patches'], embed_dim))
            self.freq_patch_size = config['frequency_size']//audio_patch_size[1]
            self.freq_embed = nn.Parameter(torch.zeros(
                1, self.freq_patch_size, config["hidden_size"]))

        self.norm = norm_layer(embed_dim)

        if self.use_mae:
            self.decoder_pos_embed_v = nn.Parameter(
                torch.zeros(1, self.num_patches_v, decoder_embed_dim))
            self.decoder_temporal_embed = nn.Parameter(
                torch.zeros(1, config['max_frames'], decoder_embed_dim))
            self.decoder_embed = nn.Linear(
                embed_dim, decoder_embed_dim, bias=True)
            self.decoder_type_embed_v = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))
            self.mask_token_v = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_norm = norm_layer(decoder_embed_dim)
            if use_audio:
                self.decoder_type_embed_a = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim))
                self.decoder_pos_embed_a = nn.Parameter(torch.zeros(
                    1, config['max_audio_patches'], decoder_embed_dim))
                self.decoder_freq_embed = nn.Parameter(
                    torch.zeros(1, self.freq_patch_size, decoder_embed_dim))
                self.mask_token_a = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim))

        self.num_frames = config["num_frames"]
        self.max_audio_patches = config['max_audio_patches']
        self.frame_masking = config["frame_masking"]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if self.use_mae:
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )

        hs = config["hidden_size"]
        self.use_text = config['use_text']
        if config['use_text']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
            )
            self.text_embeddings = BertEmbeddings(bert_config)
            self.text_embeddings.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["vam"] > 0 or config["loss_names"]["vtm"] > 0:
            self.matching_score = heads.MatchingHead(config["hidden_size"])
            self.matching_score.apply(objectives.init_weights)

        if config["loss_names"]["mae_audio"] > 0:
            self.mae_score_audio = heads.MAEHead(
                config["decoder_hidden_size"], config['audio_patch_size'][0]*config['audio_patch_size'][1])
            self.audio_patch_size = config['audio_patch_size']
            self.mae_score_audio.apply(objectives.init_weights)

        if config["loss_names"]["mae_video"] > 0:
            self.patch_size = config['patch_size']
            self.num_patches = config['video_size']//config['patch_size']
            self.mae_score_video = heads.MAEHead(
                config["decoder_hidden_size"], config['patch_size']**2*3)
            self.mae_score_video.apply(objectives.init_weights)

        # ===================== Downstream ===================== #

        if config["loss_names"]["mosei"] > 0:
            vs = 1
            self.classifier = nn.Sequential(
                heads.Pooler(config["hidden_size"]),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.classifier.apply(objectives.init_weights)

        if config["loss_names"]["moseiemo"] > 0:
            self.classifier = nn.Sequential(
                heads.Pooler(config["hidden_size"]),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 6),
            )
            self.classifier.apply(objectives.init_weights)

        if config["loss_names"]["vqa"] > 0:
            vs = config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                heads.Pooler(config["hidden_size"]),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

    def init_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        std = 0.02

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.use_audio:
            w = self.patch_embed_a.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.cls_token, std=std)
        nn.init.normal_(self.temporal_embed, std=std)
        nn.init.normal_(self.type_embed_v, std=std)
        if self.use_audio:
            nn.init.normal_(self.freq_embed, std=std)
            nn.init.normal_(self.type_embed_a, std=std)

        if self.use_mae:
            nn.init.normal_(self.decoder_type_embed_v, std=std)
            nn.init.normal_(self.decoder_temporal_embed, std=std)
            nn.init.normal_(self.mask_token_v, std=std)

            if self.use_audio:
                nn.init.normal_(self.decoder_type_embed_a, std=std)
                nn.init.normal_(self.decoder_freq_embed, std=std)
                nn.init.normal_(self.mask_token_a, std=std)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed_v", "pos_embed_a", "cls_token", "mask_token_v", "mask_token_a", "temporal_embed", "decoder_pos_embed_v", "decoder_pos_embed_a"}

    def get_span_patch(audio_spans):
        patch_span = []
        patch_indexes = []
        for i in range(len(audio_spans)):
            span_i = []
            indexes_i = []
            for span in audio_spans[i]:
                s, t = torch.round(
                    span[0]/16).cpu().numpy(), torch.round(span[1]/16).cpu().numpy()
                span_i += [[s, t]]
                indexes_i += list(range(s, t))
            patch_span += [span_i]
            patch_indexes += [indexes_i]
        return patch_span, patch_indexes

    def random_masking_audio(self, x, att_mask=None, mask_ratio=0.15, audio_spans=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        F, T = 8, L//8  # frequency, time
        if audio_spans is not None:
            patch_span, patch_indexes = self.get_span_patch(audio_spans)
            len_keep = int(L * (1 - mask_ratio))
            noise = []
            for i in range(N):
                tmp_noise = torch.rand(len(patch_span[i]), device=x.device)
                noise_i = []
                for t in range(T):
                    if t in patch_indexes[i]:
                        noise_i += [tmp_noise[i, t]]
                    else:
                        noise_i += [torch.rand(1, device=x.device)[0]+1.0]
                noise += [noise_i]
            noise = torch.tensor(noise).to(x.device)
        else:
            len_keep = int(L * (1 - mask_ratio))
            # noise in [0, 1]
            noise = torch.rand(
                N, T, device=x.device).unsqueeze(-1).repeat(1, 1, F).view(N, L)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if att_mask is not None:
            mask *= att_mask

        att_mask = torch.gather(att_mask, dim=1, index=ids_keep)
        return x_masked, mask, ids_restore, att_mask

    def random_masking(self, x, att_mask=None, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if att_mask is not None:
            mask *= att_mask

        att_mask = torch.gather(att_mask, dim=1, index=ids_keep)
        return x_masked, mask, ids_restore, att_mask

    def cat_mask(self, mask_token, x, ids_restore):
        mask_tokens = mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        return x_

    def get_patch_mask(self, x):
        """
        masks out blank regions of the audios/images.
        """
        if len(x.shape) == 5:
            x = x.mean(2)
            x = F.avg_pool2d(x, self.patch_size,
                             self.patch_size).flatten(2).flatten(1)
            x_mask = x != -1
            return x_mask
        else:
            x = x.mean(1)
            x = F.avg_pool2d(x, self.audio_patch_size,
                             self.audio_patch_size).flatten(1)
            x_mask = x != -1
            return x_mask

    def forward(self, text_ids=None, text_masks=None, audio=None, audio_masks=None, video=None, video_masks=None, mask_visual=False, use_mae=False, audio_spans=None):

        if text_ids is not None:
            text_embeds = self.text_embeddings(text_ids)
        else:
            text_embeds = None

        if audio is not None:
            x_a = self.patch_embed_a(audio)
            x_a += self.freq_embed.repeat(1, x_a.size(1)//self.freq_patch_size, 1)
            x_a += torch.repeat_interleave(self.pos_embed_a[:, :x_a.size(
                1)//self.freq_patch_size], self.freq_patch_size, dim=1)
            x_a += self.type_embed_a
            full_x_mask_a = self.get_patch_mask(audio)

        if video is not None:
            b, t, c, h, w = video.shape
            x_v = self.patch_embed_v(video.reshape(b*t, c, h, w))
            x_v = x_v.reshape(b, t * x_v.size(1), x_v.size(-1))
            frame_patch_len = x_v.size(1)//t
            x_v += self.pos_embed_v.repeat(1, t, 1)
            x_v += torch.repeat_interleave(
                self.temporal_embed[:, :self.num_frames], frame_patch_len, dim=1)
            x_v += self.type_embed_v
            full_x_mask_v = self.get_patch_mask(video)

        if mask_visual:
            if video is not None:
                x_v, mask_v, ids_restore_v, enc_x_mask_v = self.random_masking(
                    x_v, full_x_mask_v)
            if audio is not None:
                if self.frame_masking:
                    x_a, mask_a, ids_restore_a, enc_x_mask_a = self.random_masking_audio(
                        x_a, full_x_mask_a, audio_spans=audio_spans)
                else:
                    x_a, mask_a, ids_restore_a, enc_x_mask_a = self.random_masking(
                        x_a, full_x_mask_a)

                enc_mask = torch.cat([enc_x_mask_a, enc_x_mask_v], 1)
                dec_mask = torch.cat([full_x_mask_a, full_x_mask_v], 1)
                x = torch.cat([x_a, x_v], 1)
            if text_embeds is not None:
                enc_mask = torch.cat([text_masks, enc_x_mask_v], 1)
                x = torch.cat([text_embeds, x_v], 1)
                dec_mask = full_x_mask_v

        else:
            if audio is not None and video is not None:
                enc_mask = torch.cat(
                    [full_x_mask_a[:, :1], full_x_mask_a, full_x_mask_v], 1)
                x = torch.cat([self.cls_token.expand(
                    x_v.size(0), -1, -1), x_a, x_v], 1)
            elif audio is not None:
                enc_mask = full_x_mask_a
                x = x_a

            if text_embeds is not None:
                enc_mask = torch.cat(
                    [text_masks[:, :1], text_masks, full_x_mask_v], 1)
                x = torch.cat([self.cls_token.expand(
                    text_embeds.size(0), -1, -1), text_embeds, x_v], 1)

        for blk in self.blocks:
            x = blk(x, enc_mask)
        x = self.norm(x)

        if mask_visual and use_mae:
            decoder_x = self.decoder_embed(x)

            if audio is not None:
                decoder_x_a = decoder_x[:, :x_a.size(1)]
                decoder_x_a = self.cat_mask(
                    self.mask_token_a, decoder_x_a, ids_restore_a)
                decoder_x_a += self.decoder_freq_embed.repeat(
                    1, decoder_x_a.size(1)//self.freq_patch_size, 1)
                decoder_x_a += torch.repeat_interleave(self.decoder_pos_embed_a[:, :decoder_x_a.size(
                    1)//self.freq_patch_size], self.freq_patch_size, dim=1)
                decoder_x_a += self.decoder_type_embed_a
                for i, blk in enumerate(self.decoder_blocks):
                    decoder_x_a = blk(decoder_x_a)
                decoder_x_a = self.decoder_norm(decoder_x_a)
            else:
                decoder_x_a = mask_a = None

            decoder_x_v = decoder_x[:, -x_v.size(1):]
            decoder_x_v = self.cat_mask(
                self.mask_token_v, decoder_x_v, ids_restore_v)
            decoder_x_v += self.decoder_pos_embed_v.repeat(1, t, 1)
            decoder_x_v += torch.repeat_interleave(
                self.decoder_temporal_embed[:, :self.num_frames], frame_patch_len, dim=1)
            decoder_x_v += self.decoder_type_embed_v
            for blk in self.decoder_blocks:
                decoder_x_v = blk(decoder_x_v)

            decoder_x_v = self.decoder_norm(decoder_x_v)

            return None, decoder_x_a, decoder_x_v, None, mask_a, mask_v

        if text_embeds is not None:
            text_feats = x[:, 1: 1+text_embeds.size(1)]
            return x, None, None, text_feats, None, None
        else:
            return x, None, None, None, None, None


@register_model
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = TVLT(
        patch_size=16, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def mae_vit_base_patch128_dec512d8b(**kwargs):
    model = TVLT(
        patch_size=16, audio_patch_size=[2, 128], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
