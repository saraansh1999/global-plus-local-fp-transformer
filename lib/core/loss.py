import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment

def linear_combination(x, y, epsilon):
        return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' \
            else loss.sum() if reduction == 'sum' else loss


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, weight=self.weight, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class RelLoss(nn.Module):
    def __init__(self):
        super(RelLoss, self).__init__()

    def forward(self, x, target):
        n = x.size(0)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(x.float(), x.float().t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        n = target.size(0)
        dist_t = torch.pow(target, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_t = dist_t + dist_t.t()
        dist_t.addmm_(target.float(), target.float().t(), beta=1, alpha=-2)
        dist_t = dist_t.clamp(min=1e-12).sqrt()  # for numerical stability

        rel_loss = torch.mean(torch.norm(dist - dist_t, p=2))
        return rel_loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, w_kd: float = 1, w_posori: float = 1, ret_dec_intermediate=False, use_ori=True):
        super().__init__()
        self.w_kd = w_kd
        self.w_posori = w_posori
        self.use_ori = use_ori
        self.ret_dec_intermediate = ret_dec_intermediate
        print("Hungarian: ")
        print("w_kd: ", self.w_kd)
        print("w_posori", self.w_posori)
        print("use ori?", self.use_ori)
        print("inter losses?", self.ret_dec_intermediate)
        assert w_kd != 0 or w_posori != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, iter_id=-1):
        bs, num_queries = outputs["kd_tokens"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.ret_dec_intermediate:
            bs, num_queries = outputs["kd_tokens"].shape[1:3]
            out_embs = outputs["kd_tokens"][iter_id].flatten(0, 1)
            out_posori = outputs["posori_tokens"][iter_id].flatten(0, 1)
        else:
            bs, num_queries = outputs["kd_tokens"].shape[:2]
            out_embs = outputs["kd_tokens"].flatten(0, 1)
            out_posori = outputs["posori_tokens"].flatten(0, 1)

        tgt_embs = torch.cat([targets["local_emb"][i, :targets["num_mnt"][i]] for i in range(bs)])
        tgt_posori = torch.cat([targets["mnt"][i, :targets["num_mnt"][i]] for i in range(bs)])

        # removing ori dim
        if not self.use_ori:
            out_posori = out_posori[:, :2]
            tgt_posori = tgt_posori[:, :2]

        if self.w_kd != 0:
            cost_kd = torch.cdist(out_embs, tgt_embs, p=2)
        else:
            cost_kd = 0

        cost_posori = torch.cdist(out_posori, tgt_posori, p=2)

        # Final cost matrix
        C = self.w_kd * cost_kd + self.w_posori * cost_posori
        C = C.view(bs, num_queries, -1).cpu()
        sizes = tuple(targets["num_mnt"].tolist())
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class MatchThenSupervise(nn.Module):

    def __init__(self, loss_names, loss_weights, is_local, hungarian_weights, ret_dec_intermediate, hungarian_use_ori):
        super().__init__()
        self.loss_names = loss_names
        self.loss_weights = [float(i) for i in loss_weights]
        self.is_local = is_local
        self.ret_dec_intermediate = ret_dec_intermediate
        hungarian_weights = [float(i) for i in hungarian_weights]
        self.matcher = HungarianMatcher(w_kd=hungarian_weights[0], w_posori=hungarian_weights[1], ret_dec_intermediate=self.ret_dec_intermediate, use_ori=hungarian_use_ori)
        self.KDTokenLoss = nn.MSELoss()
        self.KDGlobalLoss = nn.MSELoss()
        self.PosoriLoss = nn.MSELoss()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_mnt_kd(self, outputs, targets, indices):
        if self.ret_dec_intermediate:
            loss = 0
            for lid in range(outputs['kd_tokens'].shape[0]):
                idx = self._get_src_permutation_idx(indices[lid])
                target_embs = torch.cat([t[J] for t, (_, J) in zip(targets["local_emb"], indices[lid])])
                src_embs = outputs['kd_tokens'][lid]
                src_embs = torch.cat([src_embs[bi, si].unsqueeze(0) for bi, si in zip(idx[0], idx[1])])
                loss_lid = self.KDTokenLoss(src_embs, target_embs)
                loss += loss_lid
        else:
            src_embs = outputs['kd_tokens']
            idx = self._get_src_permutation_idx(indices)
            src_embs = torch.cat([src_embs[bi, si].unsqueeze(0) for bi, si in zip(idx[0], idx[1])])

            target_embs = torch.cat([t[J] for t, (_, J) in zip(targets["local_emb"], indices)])
            loss = self.KDTokenLoss(src_embs, target_embs)

        return loss

    def loss_mnt_posori(self, outputs, targets, indices):
        if self.ret_dec_intermediate:
            loss = 0
            for lid in range(outputs['posori_tokens'].shape[0]):
                idx = self._get_src_permutation_idx(indices[lid])
                target_posori = torch.cat([t[i] for t, (_, i) in zip(targets["mnt"], indices[lid])], dim=0)
                src_posori = outputs['posori_tokens'][lid]
                src_posori = src_posori[idx]
                loss_lid = self.PosoriLoss(src_posori, target_posori)
                loss += loss_lid
        else:
            idx = self._get_src_permutation_idx(indices)
            target_posori = torch.cat([t[i] for t, (_, i) in zip(targets["mnt"], indices)], dim=0)
            src_posori = outputs['posori_tokens'][idx]
            loss = self.PosoriLoss(src_posori, target_posori)

        return loss

    def loss_mnt_pos(self, outputs, targets, indices):
        if self.ret_dec_intermediate:
            loss = 0
            for lid in range(outputs['posori_tokens'].shape[0]):
                idx = self._get_src_permutation_idx(indices[lid])
                target_posori = torch.cat([t[i] for t, (_, i) in zip(targets["mnt"], indices[lid])], dim=0)[..., :2]
                src_posori = outputs['posori_tokens'][lid]
                src_posori = src_posori[idx][..., :2]
                loss_lid = self.PosoriLoss(src_posori, target_posori)
                loss += loss_lid
        else:
            idx = self._get_src_permutation_idx(indices)
            target_posori = torch.cat([t[i] for t, (_, i) in zip(targets["mnt"], indices)], dim=0)[..., :2]
            src_posori = outputs['posori_tokens'][idx][..., :2]
            loss = self.PosoriLoss(src_posori, target_posori)

        return loss

    def loss_mnt_ori(self, outputs, targets, indices):
        if self.ret_dec_intermediate:
            loss = 0
            for lid in range(outputs['posori_tokens'].shape[0]):
                idx = self._get_src_permutation_idx(indices[lid])
                target_posori = torch.cat([t[i] for t, (_, i) in zip(targets["mnt"], indices[lid])], dim=0)[..., 2]
                src_posori = outputs['posori_tokens'][lid]
                src_posori = src_posori[idx][..., 2]
                loss_lid = self.PosoriLoss(src_posori, target_posori)
                loss += loss_lid
        else:
            idx = self._get_src_permutation_idx(indices)
            target_posori = torch.cat([t[i] for t, (_, i) in zip(targets["mnt"], indices)], dim=0)[..., 2]
            src_posori = outputs['posori_tokens'][idx][..., 2]
            loss = self.PosoriLoss(src_posori, target_posori)

        return loss

    def loss_global_kd(self, outputs, targets, indices):
        loss = self.KDGlobalLoss(outputs['kd_global'], targets['global_emb'])
        return loss

    def forward(self, outputs, targets):
        indices = None
        if self.is_local:
            if self.ret_dec_intermediate:
                indices = []
                for lid in range(outputs['posori_tokens'].shape[0]):
                    indices.append(self.matcher(outputs, targets, lid))
            else:
                indices = self.matcher(outputs, targets)

        losses = {}
        loss = 0
        for i, loss_name in enumerate(self.loss_names):
            losses[loss_name] = getattr(self, loss_name)(outputs, targets, indices)
            loss += self.loss_weights[i] * losses[loss_name]

        losses['loss_tot'] = loss

        return losses, loss


def build_criterion(config, train=True, class_counter=None):
    if config.AUG.MIXUP_PROB > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = SoftTargetCrossEntropy() \
            if train else nn.CrossEntropyLoss()
    elif config.LOSS.LABEL_SMOOTHING > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = LabelSmoothingCrossEntropy(config.LOSS.LABEL_SMOOTHING)
    elif config.LOSS.LOSS == 'softmax':
        criterion = nn.CrossEntropyLoss()
    elif config.LOSS.LOSS == 'match-then-supervise':
        criterion = MatchThenSupervise(config.LOSS.LOSS_NAMES, config.LOSS.LOSS_WEIGHTS, config.MODEL.RET_LOCAL, config.LOSS.HUNGARIAN_WEIGHTS, config.MODEL.RET_DEC_INTERMEDIATE, config.LOSS.HUNGARIAN_USE_ORI)
    else:
        raise ValueError('Unkown loss {}'.format(config.LOSS.LOSS))

    return criterion
