# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import pdb

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY

from torch_geometric.nn import GATConv, GATv2Conv, GAT


class SRSGL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout, layers):
        super().__init__()
        assert in_channels == out_channels
        self.dropout = dropout
        self.layers = layers

        for index in range(layers):
            if index + 1 <= layers:
                self.__setattr__(f'gatlayer{index+1}', GATv2Conv(in_channels, hidden_channels, heads, concat=True, dropout=dropout))
            else:
                self.__setattr__(f'gatlayer{index+1}', GATv2Conv(hidden_channels * heads, out_channels, 1, concat=False, dropout=dropout))



    def forward(self, x, edge_index):
        for index in range(self.layers):
            gat_layer = self.__getattr__(f'gatlayer{index+1}')
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = gat_layer(x, edge_index)
            if index + 1 < self.layers:
                x = F.elu(x)
        return x

@REID_HEADS_REGISTRY.register()
class EmbeddingGATHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type,
            datasets,
            gat_layers,
            gat_heads
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        self.gat = SRSGL(in_channels=feat_dim, hidden_channels=int(feat_dim/gat_heads), out_channels=feat_dim,
                         layers=gat_layers, heads=gat_heads, dropout=0.1)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.reset_parameters()
        self.num_classes = num_classes

    def reset_parameters(self) -> None:
        # self.W.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        datasets      = cfg.DATASETS.NAMES[0]
        gat_layers    = cfg.MODEL.HEADS.LAYERS
        gat_heads     = cfg.MODEL.HEADS.MULTIHEADS
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type,
            'datasets': datasets,
            'gat_layers': gat_layers,
            'gat_heads': gat_heads
        }

    def one_hot_refine(self, one_hot, img_num_ps):
        for index, value in enumerate(img_num_ps):
            start = 6 * index + value
            end = 6 * (index + 1)
            one_hot[start:end] = 0
        return one_hot

    def forward(self, features, camids=None, img_num_ps=None, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        # pdb.set_trace()

        batchsize = features.shape[0]
        features = torch.chunk(features, 6, dim=1)
        features = torch.cat(features, dim=0).squeeze()

        if len(features.shape) > 3:
            # pool_layer with resnet
            pool_feat = self.pool_layer(features)
            pool_feat = pool_feat[..., 0, 0]
        else:
            # pool_layer with Vit
            pool_feat = features


        # construction intra-group graph
        instance_target = torch.tensor([[i + 1] * 6 for i in range(batchsize)]).to(features.device).flatten()
        instance_one_hot = F.one_hot(instance_target, num_classes=batchsize + 1).double()
        instance_one_hot = self.one_hot_refine(instance_one_hot, img_num_ps)
        intra_graph_adj = torch.mm(instance_one_hot, torch.transpose(instance_one_hot, 0, 1))
        intra_graph_edge_index = torch.stack(torch.where(intra_graph_adj == 1), dim=0)

        if self.training:
            # construction all-batch graph
            person_targets = targets.repeat(6,1).T.flatten()
            person_one_hot = F.one_hot(person_targets, num_classes=self.num_classes).double()
            person_one_hot= self.one_hot_refine(person_one_hot, img_num_ps)
            person_level_adj = torch.mm(person_one_hot, torch.transpose(person_one_hot, 0, 1))

            # construction inter-group graph
            inter_graph_adj = person_level_adj - intra_graph_adj
            inter_graph_edge_index = torch.stack(torch.where(inter_graph_adj==1), dim=0)

            # # process inter-graph
            inter_feat = self.gat(pool_feat, inter_graph_edge_index)
            intra_feat = self.gat(pool_feat, intra_graph_edge_index)
            graph_feat = inter_feat + intra_feat

        else:
            # process intra-graph
            intra_feat = self.gat(pool_feat, intra_graph_edge_index)
            graph_feat = intra_feat
            # # without seperare
            # graph_feat = pool_feat

        feat_ori = pool_feat
        feat_ori = torch.stack(torch.chunk(feat_ori, 6, dim=0), dim=1)
        feat_ori_avg = feat_ori.mean(dim=1).unsqueeze(-1).unsqueeze(-1)


        # res graph
        graph_feat = graph_feat + pool_feat

        graph_feat = torch.stack(torch.chunk(graph_feat, 6, dim=0), dim=1)
        graph_feat_avg = graph_feat.mean(dim=1).unsqueeze(-1).unsqueeze(-1)


        # pool_feat = feat_after_gat
        neck_feat_ori = self.bottleneck(feat_ori_avg)
        neck_feat_ori = neck_feat_ori[..., 0, 0]
        neck_graph_feat = self.bottleneck(graph_feat_avg)
        neck_graph_feat = neck_graph_feat[..., 0, 0]
        # neck_feat_intraG = self.bottleneck(feat_intraG_avg)
        # neck_feat_intraG = neck_feat_intraG[..., 0, 0]
        # if self.training:
        #     neck_feat_interG = self.bottleneck(feat_interG_avg)
        #     neck_feat_interG = neck_feat_interG[..., 0, 0]


        # Evaluation
        # fmt: off
        # if not self.training: return neck_feat_ori
        if not self.training: return neck_graph_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat_ori, self.weight)
            # logits_intraG = F.linear(neck_feat_intraG, self.weight)
            # logits_interG = F.linear(neck_feat_interG, self.weight)
            logits_graph = F.linear(neck_graph_feat, self.weight)

        else:
            logits = F.linear(F.normalize(neck_feat_ori), F.normalize(self.weight))
            # logits_intraG = F.linear(F.normalize(neck_feat_intraG), F.normalize(self.weight))
            # logits_interG = F.linear(F.normalize(neck_feat_interG), F.normalize(self.weight))
            logits_graph = F.linear(F.normalize(neck_graph_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)
        # cls_outputs_intraG = self.cls_layer(logits_intraG.clone(), targets)
        # cls_outputs_interG = self.cls_layer(logits_interG.clone(), targets)
        cls_outputs_graph = self.cls_layer(logits_graph.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':
            feat = feat_ori_avg[..., 0, 0]
            # feat_intraG_out = feat_intraG_avg[..., 0, 0]
            # feat_interG_out = feat_interG_avg[..., 0, 0]
            feat_G = graph_feat_avg[..., 0, 0]
        elif self.neck_feat == 'after':
            feat = neck_feat_ori
            # feat_intraG_out = neck_feat_intraG
            # feat_interG_out = neck_feat_interG
            feat_G = neck_graph_feat
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            # "cls_outputs_intraG": cls_outputs_intraG,
            # "cls_outputs_interG": cls_outputs_interG,
            "cls_outputs_graph": cls_outputs_graph,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
            # "features_intraG": feat_intraG_out,
            # "features_interG": feat_interG_out,
            "features_graph": feat_G
        }
