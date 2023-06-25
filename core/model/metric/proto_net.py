# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/nips/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  booktitle = {Advances in Neural Information Processing Systems 30: Annual Conference
               on Neural Information Processing Systems 2017, December 4-9, 2017,
               Long Beach, CA, {USA}},
  pages     = {4077--4087},
  year      = {2017},
  url       = {https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html}
}
https://arxiv.org/abs/1703.05175

Adapted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch.
"""
import pdb
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random

from core.utils import accuracy
from .metric_model import MetricModel
from .. import DistillKLLoss
from core.model.loss import L2DistLoss

class ChannelAttention(nn.Module):
    def __init__(self, in_dim=640):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_dim, in_dim//16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_dim//16, in_dim, 1, bias=False))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): #[25, 640, 5, 5]

        avg_out = self.fc(self.avg_pool(x)) #[25, 640, 1, 1]
        max_out = self.fc(self.max_pool(x)) #[25, 640, 1, 1]
        out = avg_out + max_out #[25, 640, 1, 1]

        return self.sigmoid(out)

class BT(nn.Module):
    def __init__(self):
        super(BT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(64, affine=False)
        self.loss_fn2 = torch.nn.MSELoss(reduction='mean')
    
    
    def forward(self, x, N_way, N_shot):
        N, c = x.size()

        x = x.reshape(N, 1, c) 
        x = F.normalize(x, p=2, dim=1)
        x_T = x.permute(0,2,1) 
        c = x_T @ x 
        _, m, n = c.size()
        C = torch.eye(m,n).cuda().unsqueeze(0)
        loss = self.loss_fn2(c,C)

        return loss

class ChannelShuffle(nn.Module):
    def __init__(self, N_way, N_shot):
        super(ChannelShuffle, self).__init__()
        self.way = N_way
        self.shot = N_shot
    
    # mixup
    def forward(self, x, s_ca, shuffle_num=400):
        N, c, h, w = x.size()
        old_x = x
        _, replace_indexes = torch.topk(s_ca, shuffle_num, dim=1, largest=True)
        replace_indexes = replace_indexes.squeeze(-1).squeeze(-1)
        aug_x = old_x.data
        for i in range(N):
            index = np.array(random.sample(list(range(0, c)), shuffle_num))
            j = random.randint(0, N-1)
            while(i==j):
                j = random.randint(0, N-1)
            aug_x[i,replace_indexes[i],:,:].data = 0.7 * aug_x[i,replace_indexes[i],:,:].data + 0.3 * old_x[j,index,:,:].data
        
        aug_x = aug_x.reshape(self.way, -1, c, h, w)
        x = x.reshape(self.way, -1, c, h, w)
        
        aug_x = torch.cat((x, aug_x), dim=1)
        s_ca = s_ca.reshape(self.way, -1, c, 1, 1)
        s_ca = s_ca.repeat(1, 2, 1, 1, 1)

        aug_x = aug_x * s_ca
        aug_x = aug_x.reshape(-1, c, h, w)

        return aug_x

# FIXME: Add multi-GPU support
class DistillLayer(nn.Module):
    def __init__(
        self,
        emb_func,
        cls_classifier,
        is_distill,
        emb_func_path=None,
        cls_classifier_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.cls_classifier = self._load_state_dict(cls_classifier, cls_classifier_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model_state_dict)
            new_model = copy.deepcopy(model)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.emb_func is not None and self.cls_classifier is not None:
            output = self.emb_func(x)
            output = self.cls_classifier(output)

        return output

class ProtoLayer(nn.Module):
    def __init__(self):
        super(ProtoLayer, self).__init__()

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
        mode="euclidean",
    ):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c
        query_feat = query_feat.view(t, way_num * query_num, c)
        # t, w, c
        support_feat = support_feat.view(t, way_num, -1, c)
        proto_feat = torch.mean(support_feat, dim=2)

        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            ),
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            ),
        }[mode](query_feat, proto_feat)


class ProtoNet(MetricModel):
    def __init__(
        self, 
        feat_dim,
        num_class,
        gamma=1,
        alpha=1,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        cls_classifier_path=None,
        **kwargs):
        super(ProtoNet, self).__init__(**kwargs)
        
        self.feat_dim = feat_dim
        self.num_class = num_class

        self.gamma = gamma
        self.alpha = alpha

        self.is_distill = is_distill

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_class)
        # self.patch_cls_classifier = nn.Linear(self.feat_dim, self.num_class)

        self.rot_classifier = nn.Linear(self.num_class, 4)
        # self.patch_classifier = nn.Linear(self.feat_dim, 9) #patch是独立的分类头
        # self.SimSiam_classifier = prediction_MLP()
        self.ca = ChannelAttention()
        self.proto_layer = ProtoLayer()
        self.BT = BT()
        self.cs = ChannelShuffle(N_way=5, N_shot=1)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.l2_loss_func = L2DistLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self.distill_layer = DistillLayer(
            self.emb_func,
            self.cls_classifier,
            self.is_distill,
            emb_func_path,
            cls_classifier_path,
        )

        self.avgpool = nn.AvgPool2d(5, stride=1)

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        support_feat, query_feat, support_target, query_target = self.split_by_episode(image, mode=2)

        query_feat = self.emb_func(query_feat.squeeze(0),mode=2)
        support_feat = self.emb_func(support_feat.squeeze(0), mode=2)

        s_ca = self.ca(support_feat)
        support_feat = support_feat * s_ca
        # support_feat = self.cs(support_feat, s_ca)
        support_feat = self.avgpool(support_feat)
        support_feat = support_feat.view(support_feat.size(0), -1)

        q_ca = self.ca(query_feat)
        query_feat = query_feat * q_ca
        query_feat = self.avgpool(query_feat)
        query_feat = query_feat.view(query_feat.size(0), -1)
        
        query_feat = query_feat.unsqueeze(0)
        support_feat = support_feat.unsqueeze(0)

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.view(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        support_feat, query_feat, support_target, query_target = self.split_by_episode(images, mode=2)
        
        query_feat = self.emb_func(query_feat.squeeze(0),mode=2)
        support_feat = self.emb_func(support_feat.squeeze(0), mode=2)

        s_ca = self.ca(support_feat)
        # support_feat = support_feat * s_ca
        support_feat = self.cs(support_feat, s_ca)
        support_feat = self.avgpool(support_feat)
        support_feat = support_feat.view(support_feat.size(0), -1)

        q_ca = self.ca(query_feat)
        query_feat = query_feat * q_ca
        query_feat = self.avgpool(query_feat)
        query_feat = query_feat.view(query_feat.size(0), -1)

        query_feat = query_feat.unsqueeze(0)
        support_feat = support_feat.unsqueeze(0)

        loss_BT = 0.1 * self.BT(support_feat.squeeze(0), self.way_num, self.shot_num)

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)

        loss_cr = self.ce_loss_func(output, query_target.view(-1))
        loss = loss_BT + loss_cr
        # loss = loss_cr
        
        acc = accuracy(output, query_target.view(-1))

        # return output, acc, loss
        return output, acc, loss, loss_cr, loss_BT
