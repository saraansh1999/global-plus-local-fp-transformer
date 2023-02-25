import copy
from timm.models.layers import trunc_normal_
from .registry import register_model
from .cls_cvt import *
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_embs=50, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
        i = torch.arange(w, device=x.device).long()
        j = torch.arange(h, device=x.device).long()
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], h*w, -1)
        return pos              # BS x HW x D

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                posori_vals: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            output = torch.stack(intermediate)

        return output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ConvolutionalVisionTransformerWithTokenOut(ConvolutionalVisionTransformer):
    def __init__(self, in_chans, num_classes, act_layer, norm_layer, init, spec):
        super().__init__(in_chans, num_classes, act_layer, norm_layer, init, spec, True)

    def forward(self, x):
        cls, x = self.forward_features(x)
        kd_x = self.head(cls)

        return cls, x, kd_x

class CVTDETR(nn.Module):
    def __init__(self,
                 pretrained,
                 pre_mode,
                 global_emb_sz,
                 local_emb_sz,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None,
                 max_mnt=50,
                 token_norm=False,
                 ret_local=False,
                 ret_global=False,
                 ret_dec_intermediate=False,
                 afis_supervision=False):
        super().__init__()
        self.cvt = ConvolutionalVisionTransformerWithTokenOut(
                 in_chans=in_chans,
                 num_classes=num_classes,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 init=init,
                 spec=spec)

        # change/add head
        if pre_mode == "imagenet":
            sd = torch.load(pretrained)
            nsd = self.cvt.state_dict()
            for k,v in sd.items():
                if k in nsd.keys():
                    nsd[k] = v
            self.cvt.load_state_dict(nsd)
            hidden_sz = self.cvt.dim_embed
            self.cvt.head = nn.Linear(hidden_sz, global_emb_sz)
            trunc_normal_(self.cvt.head.weight, std=0.02)
        else:
            hidden_sz = self.cvt.dim_embed
            self.cvt.head = nn.Linear(hidden_sz, global_emb_sz)
            

        self.ret_global = ret_global

        self.ret_local = ret_local
        if self.ret_local:
            self.pos_encoder = PositionEmbeddingLearned(num_pos_feats=hidden_sz // 2)

            self.num_queries = max_mnt
            self.query_encoder = nn.Embedding(self.num_queries, hidden_sz)

            self.ret_dec_intermediate = ret_dec_intermediate
            decoder_layer = TransformerDecoderLayer(hidden_sz, 8, 1024, spec['DROP_RATE'][-1], "relu")
            decoder_norm = nn.LayerNorm(hidden_sz)
            self.decoder = TransformerDecoder(decoder_layer, spec['NUM_DEC'], decoder_norm, return_intermediate=ret_dec_intermediate)

            self.afis_supervision = afis_supervision
            self.kd_linear = nn.Linear(hidden_sz, local_emb_sz)
            self.token_norm = token_norm

            posori_mlp = MLP(hidden_sz, hidden_sz, 3, 3)               # x, y, ori
            nn.init.constant_(posori_mlp.layers[-1].weight.data, 0)
            nn.init.constant_(posori_mlp.layers[-1].bias.data, 0)
            self.posori_mlp = posori_mlp

        if pre_mode == 'same':
            sd = torch.load(pretrained)
            nsd = self.state_dict()
            for k,v in sd.items():
                if k in nsd.keys():
                    nsd[k] = v
                    print(k, "loaded")
                else:
                    print(k, "not loaded")
            self.load_state_dict(nsd)

    def forward(self, x):
        cls, tokens, emb = self.cvt(x)
        emb = F.normalize(emb, p=2.0, dim=-1)
        bs = tokens.shape[0]

        if self.ret_local:
            pos_embed = self.pos_encoder(tokens).permute(1, 0, 2)                           # tokens x BS x hidden_sz
            query_embed = self.query_encoder.weight.unsqueeze(1).repeat(1, bs, 1)           # num_queries x BS x hidden_sz

            tgt = torch.zeros_like(query_embed)                                     # num_queries x BS x hidden_sz
            tokens = tokens.permute(1, 0, 2)                                        # tokens x BS x hidden_sz

            decoded_tokens = self.decoder(tgt, tokens, pos=pos_embed, query_pos=query_embed)               # (layers x) num_queries x BS x hidden_sz
            if self.ret_dec_intermediate:
                decoded_tokens = decoded_tokens.permute(0, 2, 1, 3)
            else:
                decoded_tokens = decoded_tokens.permute(1, 0, 2)

            posori_tokens = self.posori_mlp(decoded_tokens).sigmoid()       # BS x num_queries x 3

            if self.afis_supervision:
                kd_tokens = self.kd_linear(decoded_tokens)                      # BS x num_queries x local_emb_sz
            else:
                kd_tokens = decoded_tokens
            if self.token_norm:
                kd_tokens = F.normalize(kd_tokens, p=2.0, dim=-1)

        out = {}
        if self.ret_local:
            out['kd_tokens'] = kd_tokens
            out['posori_tokens'] = posori_tokens
        if self.ret_global:
            out['kd_global'] = emb

        return out

@register_model
def get_cls_model(config, **kwargs):
    msvit_spec = config.MODEL.SPEC
    msvit = CVTDETR(
        config.MODEL.PRETRAINED,
        config.MODEL.PRE_MODE,
        config.MODEL.GLOBAL_EMB_SZ,
        config.MODEL.TOKEN_EMB_SZ,
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec,
        max_mnt=config.DATASET.MAX_MNT,
        token_norm=config.MODEL.TOKEN_NORM,
        ret_local=config.MODEL.RET_LOCAL,
        ret_global=config.MODEL.RET_GLOBAL,
        ret_dec_intermediate=config.MODEL.RET_DEC_INTERMEDIATE,
        afis_supervision=config.MODEL.AFIS_SUPERVISION
    )

    return msvit
