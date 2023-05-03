"""
VisTR model and criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (VisTRsegm, PostProcessSegm, VisTRsegmWithEarlyExit,VisTRsegmWithEarlyExitAdaptive,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .transformer import build_transformer_with_early_exit

from typing import List, Optional
from torch import Tensor
from .dcn.deform_conv import DeformConv

class VisTR(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame, 
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        pos = pos[-1]
        src, mask = features[-1].decompose()
        
        src_proj = self.input_proj(src)
        n,c,h,w = src_proj.shape
        assert mask is not None
        src_proj = src_proj.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(0,2,1,3,4).flatten(-2)
        mask = mask.reshape(n//self.num_frames, self.num_frames, h*w)
        pos = pos.permute(0,2,1,3,4).flatten(-2)
        hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)[0]

        if len(hs.shape) == 4:
            # the case of intermediate_output=True
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        else:
            # the case of intermediate_output=False
            # hs has a shape of [15, 384, 1]
            # need to reshape to [1, 15, 384] for class_embed and bbox_embed
            hs = hs.permute(2,0,1)
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    
class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)#, bias=False)
        self.dcn = DeformConv(inter_dims[3],inter_dims[4], 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # dcn for the last layer
        offset = self.conv_offset(x)
        x = self.dcn(x,offset)
        x = self.gn5(x)
        x = F.relu(x)
        return x
    
# Reimplement by following the orig order
class VisTRWithEarlyExitOld(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 early_exit_layer, 
                 num_classes, 
                 num_frames, 
                 num_queries, 
                 aux_loss=False, 
                 enable_segm=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame, 
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.early_exit_layer = early_exit_layer
        
        self.enable_segm = enable_segm
        
        self.num_decoder_layers = transformer.num_decoder_layers
        hidden_dim = transformer.d_model
        nheads = transformer.nhead

        self.hidden_dim = hidden_dim
        
        ######### added for early exit #########
        self.class_embeds = nn.ModuleList(nn.Linear(hidden_dim, num_classes + 1) for _ in range(self.num_decoder_layers))
        self.bbox_embeds = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(self.num_decoder_layers))
        ########################################
                    
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        self.bbox_attention_layers = nn.ModuleList(MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0) \
                                                   for _ in range(self.num_decoder_layers))
        
        self.mask_head_layers = nn.ModuleList(MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim) \
                                                    for _ in range(self.num_decoder_layers))
        
        self.insmask_head_layers = nn.ModuleList(nn.Sequential(
                                nn.Conv3d(24,12,3,padding=2,dilation=2),
                                nn.GroupNorm(4,12),
                                nn.ReLU(),
                                nn.Conv3d(12,12,3,padding=2,dilation=2),
                                nn.GroupNorm(4,12),
                                nn.ReLU(),
                                nn.Conv3d(12,12,3,padding=2,dilation=2),
                                nn.GroupNorm(4,12),
                                nn.ReLU(),
                                nn.Conv3d(12,1,1)) for _ in range(self.num_decoder_layers))

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        
        bs = features[-1].tensors.shape[0] # for segmentation

        pos = pos[-1]
        src, mask = features[-1].decompose()
        src_proj = self.input_proj(src)
        n,c,h,w = src_proj.shape

        # Trying to make sure enable_segm does not affect ffn
        # the variable names are slightly different when enable_segm is on and off 
        if self.enable_segm:
            n,c,s_h,s_w = src_proj.shape
            bs_f = bs // self.num_frames # for segmentation

        assert mask is not None
                
        src_proj = src_proj.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(0,2,1,3,4).flatten(-2)
        mask = mask.reshape(n//self.num_frames, self.num_frames, h*w)
        pos = pos.permute(0,2,1,3,4).flatten(-2)

        ###### TODO: hs should have multiple objects without using the intermediate flag
        ###### each object corresponds to a decoder layer output
        other_hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)[0]
        # hs, _, other_hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)
        ###### Calculate each decoder layer's output

        if self.enable_segm:
            other_hs, memory = self.transformer(src_proj, mask, self.query_embed.weight, pos)
            for i in range(3):
                _,c_f,h,w = features[i].tensors.shape
                features[i].tensors = features[i].tensors.reshape(bs_f, self.num_frames, c_f, h,w)
                n_f = self.num_queries // self.num_frames

        res = []

        # if self.early_exit_layer > 0:
        assert self.early_exit_layer > 0, "early_exit_layer not correct"
        for layer in range(self.early_exit_layer):
            other_hs[layer] = other_hs[layer].permute(2,0,1)
            outputs_class = self.class_embeds[layer](other_hs[layer])
            assert outputs_class.shape == torch.Size([1, 15, 42]), f"outputs_class with wrong shape {outputs_class.shape}"
            outputs_coord = self.bbox_embeds[layer](other_hs[layer]).sigmoid()
            assert outputs_coord.shape == torch.Size([1, 15, 4]), f"outputs_coord with wrong shape {outputs_coord.shape}"
            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)                    

            if self.enable_segm:
                # image level processing using box attention 
                outputs_seg_masks = []
                for i in range(self.num_frames):
                    hs_f = other_hs[layer][:,i*n_f:(i+1)*n_f,:]
                    memory_f = memory[:,:,i,:].reshape(bs_f, c, s_h,s_w)
                    mask_f = mask[:,i,:].reshape(bs_f, s_h,s_w)
                    bbox_mask_f = self.bbox_attention_layers[layer](hs_f, memory_f, mask=mask_f)
                    seg_masks_f = self.mask_head_layers[layer](memory_f, bbox_mask_f, [features[2].tensors[:,i], features[1].tensors[:,i], features[0].tensors[:,i]])
                    outputs_seg_masks_f = seg_masks_f.view(bs_f, n_f, 24, seg_masks_f.shape[-2], seg_masks_f.shape[-1])
                    outputs_seg_masks.append(outputs_seg_masks_f)
                frame_masks = torch.cat(outputs_seg_masks,dim=0)
                
                outputs_seg_masks = []

                # instance level processing using 3D convolution
                for i in range(frame_masks.size(1)):
                    mask_ins = frame_masks[:,i].unsqueeze(0)
                    mask_ins = mask_ins.permute(0,2,1,3,4)
                    outputs_seg_masks.append(self.insmask_head_layers[layer](mask_ins))
                outputs_seg_masks = torch.cat(outputs_seg_masks,1).squeeze(0).permute(1,0,2,3)
                outputs_seg_masks = outputs_seg_masks.reshape(1,self.num_queries,outputs_seg_masks.size(-2),outputs_seg_masks.size(-1))
                out["pred_masks"] = outputs_seg_masks

            res.append(out)

        return res

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class VisTRWithEarlyExit(nn.Module):
    """ This is the VisTR module that performs video object detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 early_exit_layer, 
                 num_classes, 
                 num_frames, 
                 num_queries, 
                 aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame, 
                         thus 360 queries for 36 frames.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        nheads = transformer.nhead
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        ######### added for early exit #########
        self.early_exit_layer = early_exit_layer
        self.num_decoder_layers = transformer.num_decoder_layers
        self.class_embeds = nn.ModuleList(nn.Linear(hidden_dim, num_classes + 1) for _ in range(self.num_decoder_layers))
        self.bbox_embeds = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(self.num_decoder_layers))
        ########################################
                    

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        pos = pos[-1]        
        bs = features[-1].tensors.shape[0] # for segmentation
        src, mask = features[-1].decompose()
        src_proj = self.input_proj(src)
        n,c,h,w = src_proj.shape

        assert mask is not None
                
        src_proj = src_proj.reshape(n//self.num_frames, self.num_frames, c, h, w).permute(0,2,1,3,4).flatten(-2)
        mask = mask.reshape(n//self.num_frames, self.num_frames, h*w)
        pos = pos.permute(0,2,1,3,4).flatten(-2)

        ###### TODO: hs should have multiple objects without using the intermediate flag
        ###### each object corresponds to a decoder layer output
        other_hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)[0]
        # hs, _, other_hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)
        ###### Calculate each decoder layer's output

        res = []
        assert self.early_exit_layer > 0, "early_exit_layer not correct"
        for layer in range(self.early_exit_layer):
            other_hs[layer] = other_hs[layer].permute(2,0,1)
            outputs_class = self.class_embeds[layer](other_hs[layer])
            assert outputs_class.shape == torch.Size([1, 15, 42]), f"outputs_class with wrong shape {outputs_class.shape}"
            outputs_coord = self.bbox_embeds[layer](other_hs[layer]).sigmoid()
            assert outputs_coord.shape == torch.Size([1, 15, 4]), f"outputs_coord with wrong shape {outputs_coord.shape}"
            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)                    
            res.append(out)

        return res

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    """ This class computes the loss for VisTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], split=False).decompose()
        target_masks = target_masks.to(src_masks)
        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        try: 
            src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
            src_masks = src_masks[:, 0].flatten(1)
            target_masks = target_masks[tgt_idx].flatten(1)
        except:
            src_masks = src_masks.flatten(1)
            target_masks = src_masks.clone()
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file == "ytvos":
        num_classes = 41
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = VisTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = VisTRsegm(model)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
    return model, criterion, postprocessors

def build_early_exit(args):
    if args.dataset_file == "ytvos":
        num_classes = 41
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer_with_early_exit(args)

    model = VisTRWithEarlyExit(
        backbone,
        transformer,
        early_exit_layer=args.early_exit_layer,
        num_classes=num_classes,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss)

    if args.masks:
        model = VisTRsegmWithEarlyExit(model, args.early_exit_layer)
        # model = VisTRsegmWithEarlyExitAdaptive(model, args.early_exit_layer)


    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
    return model, criterion, postprocessors
