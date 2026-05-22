import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed
from util.misc import inverse_sigmoid
import torch.nn.functional as F


def select_topk_in_nxn_blocks(class_score, n, k):
    """
    GPU优化版本：将特征图分成n x n块，选择每块中最大的k个值

    Args:
        class_score: tensor of shape [B, H, W]
        n: 块大小，例如3表示3x3块
        k: 每个块中选择的最大值数量

    Returns:
        mask: tensor of shape [B, H, W] (bool类型，选中为True，未选中为False)
    """
    B, H, W = class_score.shape
    assert k <= n * n, f"k({k})不能超过块内元素数量({n * n})"
    device = class_score.device

    # 1. 计算填充量并填充到n的倍数
    pad_h = (n - H % n) % n
    pad_w = (n - W % n) % n

    if pad_h > 0 or pad_w > 0:
        # 使用最小值填充
        fill_value = float(class_score.min().item())
        # 注意：对3D张量使用F.pad时，填充格式为(左, 右, 上, 下)
        x_padded = F.pad(class_score, (0, pad_w, 0, pad_h), value=fill_value)
    else:
        x_padded = class_score

    Hp, Wp = x_padded.shape[1], x_padded.shape[2]
    x_blocks = x_padded.view(B, Hp // n, n, Wp // n, n)# 重塑为块状结构 [B, Hp//n, n, Wp//n, n]
    x_blocks = x_blocks.permute(0, 1, 3, 2, 4).contiguous()# 调整维度顺序并展平最后两个空间维度 [B, Hp//n, Wp//n, n*n]
    x_blocks_flat = x_blocks.view(B, Hp // n, Wp // n, n * n)
    topk_values, topk_indices = torch.topk(x_blocks_flat, k=k, dim=-1) # 在每个n x n块中选择topk值
    mask_blocks = torch.zeros(B, Hp // n, Wp // n, n * n, dtype=torch.bool, device=device) # 使用scatter创建bool mask
    mask_blocks = mask_blocks.scatter(-1, topk_indices, 1.0)# 使用scatter将选中的位置设为True
    mask_blocks = mask_blocks.bool()# 转换为布尔类型
    mask_blocks_nxn = mask_blocks.view(B, Hp // n, Wp // n, n, n)# 将mask重塑回原始空间布局
    mask_blocks_nxn = mask_blocks_nxn.permute(0, 1, 3, 2, 4).contiguous()
    mask_padded = mask_blocks_nxn.view(B, Hp, Wp)
    mask = mask_padded[:, :H, :W]# 裁剪回原始尺寸

    return mask


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out

class DRPSTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        val_img_path = None,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        self.val_img_path = val_img_path

        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.two_stage_num_proposals_o2m = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.tgt_embed_o2m = nn.Embedding(self.two_stage_num_proposals_o2m, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.encoder_class_head_o2m = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head_o2m = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.encoder.enhance_mcsp = self.encoder_class_head#编码器的类别增强头，指向分类头。

        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)#掩码预测器，用于显著性分数。

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        nn.init.normal_(self.tgt_embed_o2m.weight)
        # initialize encoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        nn.init.constant_(self.encoder_class_head_o2m.bias, bias_value)
        # initiailize encoder regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        nn.init.constant_(self.encoder_bbox_head_o2m.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head_o2m.layers[-1].bias, 0.0)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query,
        noised_box_query,
        attn_mask,
    ):


        # get input for encoder将多尺度特征、mask、位置编码展平成一维序列。
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)

        #生成编码器输出的候选框（proposal）和特征。
        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        # 计算每个特征层需要保留的前景token数量
        #对每个特征层的 mask 取反，得到前景区域的掩码
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        #统计每个 batch 在每个特征层的有效 token 数（即有效像素点数），shape 为 (batch, num_levels)。
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        #按照每层的 level_filter_ratio计算每层要保留的前景 token 数量，并取整，shape 仍为 (batch, num_levels)。
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        #取所有 batch 中每层最大的前景 token 数，shape 为 (num_levels,)。用于后续统一处理每层的最大前景数。
        level_token_nums = focus_token_nums.max(0)[0]
        # 计算每个 batch 总共要保留的前景 token 数量，shape 为 (batch,)。
        focus_token_nums = focus_token_nums.sum(-1)

        # from high level to low level从高层到低层遍历特征层，收集每层的前景分数和索引。
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            #计算当前层在flatten序列中的起止索引，提取该层的特征和mask。
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            mask = mask_flatten[:, start_index:end_index]
            # update the memory using the higher-level score_prediction
            if level_idx != spatial_shapes.shape[0] - 1:#如果不是最高层
                #将上一层的显著性分数上采样到当前层空间大小，并加权融合到当前层特征上（显著性增强）。
                upsample_score = torch.nn.functional.interpolate(#双线性插值法上采样
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                upsample_score = upsample_score.transpose(1, 2)
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory)#用掩码预测器预测当前层每个token的显著性分数。
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())#对背景填充为最小分数。[B,N]
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])#调整score形状[B,-1,H,W]

            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1)
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)
        #合并所有层的前景分数和索引，并按分数排序。
        selected_score = torch.cat(selected_score[::-1], 1)#反转后拼接，前景分数按从低层到高层的顺序
        index = torch.sort(selected_score, dim=1, descending=True)[1]#每个batch的所有前景分数从大到小排序，得到排序后的索引
        #得到全局分数从高到低的前景token索引
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)#同样将所有层的前景token索引拼接，然后根据分数排序的index重新排列

        # create layer-wise filtering
        #按编码层比例进一步筛选前景token索引。
        #合并显著性分数，填充mask位置为最小值。
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        #按照每一层的 layer_filter_ratio计算每层最终要保留的前景 token 数量
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        #对每一层，保留前 r 个前景 token 索引，得到每层的前景 token 索引列表。
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]
        salience_score = salience_score[::-1]#保证顺序与 selected_inds 一致（从低层到高层）
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        foreground_score = foreground_score.masked_fill(mask_flatten, foreground_score.min())

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            foreground_score=foreground_score,
            focus_token_nums=focus_token_nums,
            foreground_inds=selected_inds,
        )

        if self.neck is not None:#通常用于特征融合或增强
            feat_unflatten = memory.split(spatial_shapes.prod(-1).unbind(), dim=1)
            feat_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1, self.embed_dim, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            feat_unflatten = list(self.neck(feat_unflatten).values())
            memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feat_unflatten], dim=1)

        # get encoder output, classes and coordinates生成编码器输出的类别和坐标预测。
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        topk_index = self.drps_score(enc_outputs_class, spatial_shapes, foreground_score,
                                       topk=self.two_stage_num_proposals).unsqueeze(-1)#[B, N]

        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))
        # 生成解码器的参考点和目标嵌入。
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)
        #如果有去噪标签和框，进行拼接。
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        #o2m:
        if self.training:
            enc_outputs_class_o2m = self.encoder_class_head_o2m(output_memory)
            enc_outputs_coord_o2m = self.encoder_bbox_head_o2m(output_memory) + output_proposals
            enc_outputs_coord_o2m = enc_outputs_coord_o2m.sigmoid()
            # 选取topk分数最高的类别和坐标，并做NMS去重。
            topk_index = self.drps_score(enc_outputs_class_o2m, spatial_shapes, foreground_score,
                                           topk=self.two_stage_num_proposals_o2m).unsqueeze(-1)  # [B, N]
            enc_outputs_class_o2m = enc_outputs_class_o2m.gather(1, topk_index.expand(-1, -1, self.num_classes))
            enc_outputs_coord_o2m = enc_outputs_coord_o2m.gather(1, topk_index.expand(-1, -1, 4))
            # 生成解码器的参考点和目标嵌入。
            reference_points_o2m = enc_outputs_coord_o2m.detach()
            target_o2m = self.tgt_embed_o2m.weight.expand(multi_level_feats[0].shape[0], -1, -1)
        else:
            enc_outputs_class_o2m = None
            enc_outputs_coord_o2m = None
            target_o2m = None
            reference_points_o2m = None

        # decoder调用解码器，输出最终的类别和坐标预测。
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
            o2m=False,
        )

        if self.training:
            # decoder调用解码器，输出最终的类别和坐标预测。
            outputs_classes_o2m, outputs_coords_o2m = self.decoder(
                query=target_o2m,
                value=memory,
                key_padding_mask=mask_flatten,
                reference_points=reference_points_o2m,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                attn_mask=None,
                o2m=True,
            )
        else:
            outputs_classes_o2m = None
            outputs_coords_o2m = None


        return (outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score,
                outputs_classes_o2m, outputs_coords_o2m, enc_outputs_class_o2m, enc_outputs_coord_o2m)

    @torch.no_grad()
    def drps_score(self, enc_outputs_class, spatial_shapes, fore_score, topk, n=2, k=1):
        enc_cls_score = enc_outputs_class.max(-1)[0]#[B,N]
        enc_cls_score_split = enc_cls_score.split(spatial_shapes.prod(-1).unbind(), dim=1)

        mask_cls_score = []
        for i, (feat_map, spatial_shape) in enumerate(zip(enc_cls_score_split, spatial_shapes)):
            # 获取特征图的形状
            h, w = spatial_shape
            # 重塑特征图
            feat_reshaped = feat_map.reshape(-1, h, w)
            mask = select_topk_in_nxn_blocks(feat_reshaped, n=n, k=k)
            mask = mask.reshape(-1, h * w)
            mask_cls_score.append(mask)
        score_min = enc_outputs_class.min().item()

        mask_cls_score_flatten = torch.cat(mask_cls_score, dim=1)
        enc_cls_score[mask_cls_score_flatten == 0] = score_min

        # 选取topk分数最高的类别和坐标
        topk_scores, topk_index = torch.topk(enc_cls_score, topk, dim=1)
        return topk_index

class DRPSTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
        # focus parameter
        topk_sa=300,#前景token筛选数量
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk_sa = topk_sa

        # pre attention
        self.pre_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.pre_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.pre_attention.out_proj.weight)
        # initilize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        #静态方法：如果有位置编码，则加到输入上。
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        value,  # focus parameter
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
        # focus parameter
        mc_score=None,
    ):
        #计算每个token的最大类别分数与前一层显著性分数的乘积，作为前景分数。
        #选出topk个分数最大的token索引，并扩展成与特征维度一致的shape
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]#选取topk前景token索引
        select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)#扩展成 [batch, topk, embed_dim]
        #根据索引提取topk前景token及其位置编码。
        select_tgt = torch.gather(query, 1, select_tgt_index)
        select_pos = torch.gather(query_pos, 1, select_tgt_index)
        query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
        tgt2 = self.pre_attention(
            query_with_pos,
            key_with_pos,
            select_tgt,
        )[0]
        select_tgt = select_tgt + self.pre_dropout(tgt2)
        select_tgt = self.pre_norm(select_tgt)
        #用scatter把更新后的前景token写回原序列，未被选中的token保持不变。
        query = query.scatter(1, select_tgt_index, select_tgt)

        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query

class DRPSTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6, max_num_embedding=200):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        # learnt background embed for prediction定义一个可学习的位置编码，用于为背景token生成嵌入
        #self.background_embedding = PositionEmbeddingLearned(max_num_embedding, num_pos_feats=self.embed_dim // 2)
        self.alpha = 0.5
        self.beta = 1.5
        self.gama = 1.0

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()


    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        #静态方法，生成每个token的空间参考点（归一化坐标），用于多尺度注意力。
        #对每个特征层，生成网格坐标并归一化，最后拼接成所有token的参考点
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
        self,
        query,
        spatial_shapes,#多尺度特征的空间形状
        level_start_index,
        valid_ratios,#每个特征层的有效区域缩放的比例
        query_pos=None,
        query_key_padding_mask=None,
        # salience input
        foreground_score=None,
        focus_token_nums=None,
        foreground_inds=None,
    ):
        #生成参考点，并保存原始参考点和位置编码。
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query
        for layer_id, layer in enumerate(self.layers):
            #取出当前层的前景token索引（shape: [batch, 前景数]），扩展成 [batch, 前景数, embed_dim]，用于后续gather操作
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1)
            enc_outputs_class = self.enhance_mcsp(output)

            #从上层输出的所有token中，提取当前层所有前景token的特征和位置编码
            query = torch.gather(output, 1, inds_for_query.expand(-1, -1, self.embed_dim))
            query_pos = torch.gather(ori_pos, 1, inds_for_query.expand(-1, -1, self.embed_dim))
            score_tgt = torch.gather(enc_outputs_class, 1, inds_for_query.expand(-1, -1, enc_outputs_class.shape[-1]))
            score_tgt = F.normalize(score_tgt.max(-1)[0], dim=-1) * self.alpha

            #提取当前层所有前景token的显著性分数（salience score）
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            foreground_pre_layer = F.normalize(foreground_pre_layer, dim=-1) * self.beta

            drps_score = self.drps(enc_outputs_class, spatial_shapes)
            query_drps = torch.gather(drps_score, 1, foreground_inds[layer_id])
            query_drps = F.normalize(query_drps, dim=-1) * self.gama

            mc_score = score_tgt + foreground_pre_layer + query_drps
            #print(f"alpha: {self.alpha[layer_id].item():.4f}, beta: {self.beta[layer_id].item():.4f}, gama: {self.gama[layer_id].item():.4f}")
            #根据前景索引提取参考点。[batch, 前景数, s, p]
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            #计算前景token的类别分数（score_tgt）。通常是编码器分类头输出
            #只对前景token进行编码（layer前向），其余token保持不变
            query = layer(
                query,
                query_pos,
                value,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
                mc_score=mc_score,
            )
            #用scatter操作将更新后的前景token写回原序列，未被选中的token保持原值。
            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):#对每个batch
                #只取有效的前景token索引（去掉pad部分）
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                #只取有效的前景token特征（去掉pad部分）
                query_no_pad = query[i][:focus_token_nums[i]]
                #用 scatter 操作把更新后的前景token特征写回原序列（output），未被选中的token保持原值。
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)#最后将所有batch拼接成新的output，作为下一层输入

        return output

    @torch.no_grad()
    def drps(self, enc_outputs_class, spatial_shapes, n=2, k=1):
        enc_cls_score = enc_outputs_class.max(-1)[0]#[B,N]
        enc_cls_score_split = enc_cls_score.split(spatial_shapes.prod(-1).unbind(), dim=1)
        mask_cls_score = []
        for i, (feat_map, spatial_shape) in enumerate(zip(enc_cls_score_split, spatial_shapes)):
            # 获取特征图的形状
            h, w = spatial_shape
            # 重塑特征图
            feat_reshaped = feat_map.reshape(-1, h, w)
            mask = select_topk_in_nxn_blocks(feat_reshaped, n=n, k=k).reshape(-1, h * w)
            mask_cls_score.append(mask)
        mask_cls_score_flatten = torch.cat(mask_cls_score, dim=1)
        score_min = enc_outputs_class.min().item()
        enc_cls_score[mask_cls_score_flatten == 0] = score_min
        return enc_cls_score

class DRPSTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # # self attention o2m
        # self.self_attn_o2m = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        # self.dropout2_o2m = nn.Dropout(dropout)
        # self.norm2_o2m = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)



        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
        o2m=False,
    ):
        # if not o2m:
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        # else:
        #     # self attention o2m
        #     query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        #     query2 = self.self_attn_o2m(
        #         query=query_with_pos,
        #         key=key_with_pos,
        #         value=query,
        #         attn_mask=self_attn_mask,
        #     )[0]
        #     query = query + self.dropout2_o2m(query2)
        #     query = self.norm2_o2m(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn

        query = self.forward_ffn(query)


        return query

class DRPSTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
        o2m=False,
    ):
        outputs_classes = []
        outputs_coords = []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
                o2m = o2m,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
