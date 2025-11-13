"""
YOLOv12-like single-file implementation (PyTorch).
Includes:
 - model architecture (backbone + neck + head)
 - anchor & grid generation
 - prediction decoding
 - simple target assignment
 - loss (BCE obj + BCE class + CIoU bbox)
 - non-max suppression (NMS)
 - helper training/validation step functions
 - wrapper exposing .model for compatibility with existing code expecting .model
"""

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utility functions
# -------------------------
def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [xc,yc,w,h] to [x1,y1,x2,y2]. Boxes shape (...,4)"""
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """IoU between two sets of boxes in xyxy format.
    box1: (N,4), box2: (M,4) returns (N,M) IoU
    """
    # ensure shape
    if box1.ndim == 1:
        box1 = box1.unsqueeze(0)
    if (box2.ndim == 1):
        box2 = box2.unsqueeze(0)
    N = box1.shape[0]
    M = box2.shape[0]

    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N,M,2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (N,M,2)

    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + eps
    return inter / union


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Complete IoU (CIoU) loss between predicted and target boxes.
    Input format: both in xyxy (x1,y1,x2,y2), shapes (N,4)
    Returns per-box loss (not averaged)
    """
    # IoU
    iou = box_iou(pred_boxes, target_boxes).diag()  # if same order, diagonal
    # Centers
    px = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    py = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    tx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    ty = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    # center distance squared
    center_dist2 = (px - tx) ** 2 + (py - ty) ** 2

    # enclosing box
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enc_w = (enc_x2 - enc_x1).clamp(min=eps)
    enc_h = (enc_y2 - enc_y1).clamp(min=eps)
    c2 = enc_w ** 2 + enc_h ** 2 + eps

    # aspect ratio term
    w_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps)
    h_pred = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)
    w_tgt = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps)
    h_tgt = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)

    v = (4 / (math.pi ** 2)) * ((torch.atan(w_tgt / h_tgt) - torch.atan(w_pred / h_pred)) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    loss = 1 - iou + (center_dist2 / c2) + alpha * v
    return loss


# -------------------------
# Basic building blocks
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, groups=1, act=True):
        super().__init__()
        # Auto-pad for k=3
        if k == 3:
            p = 1
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels
        self.layers = nn.Sequential(
            ConvBNAct(channels, hidden, k=1, s=1, p=0),
            ConvBNAct(hidden, channels, k=3, s=1, p=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class C2f(nn.Module):
    """Stack of Residual blocks (C2f style)"""
    def __init__(self, channels, num_blocks=1):
        super().__init__()
        self.blocks = nn.Sequential(*[Residual(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


class SPPF(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        c_ = c // 2
        self.cv1 = ConvBNAct(c, c_, k=1, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNAct(c_ * 4, c, k=1, s=1, p=0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# -------------------------
# Detection head and model
# -------------------------
class DetectHead(nn.Module):
    """
    YOLO detection head producing three-scale predictions.
    Output for each scale: (B, anchors*(5+nc), H, W)
    We'll decode using decode_predictions().
    """
    def __init__(self, in_channels: List[int], num_classes: int, anchors: List[List[Tuple[int, int]]], strides: List[int]):
        super().__init__()
        assert len(in_channels) == len(anchors) == len(strides)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_outputs = 5 + num_classes
        self.na = [len(a) for a in anchors]
        self.anchors = anchors
        self.strides = strides

        # conv layers to map features --> (na*(5+nc))
        self.m = nn.ModuleList([
            nn.Conv2d(c, self.na[i] * self.num_outputs, kernel_size=1, stride=1, padding=0)
            for i, c in enumerate(in_channels)
        ])

    def forward(self, feats: List[torch.Tensor]):
        outs = []
        for i, x in enumerate(feats):
            out = self.m[i](x)
            outs.append(out)
        return outs

    def inference_decode(self, outs: List[torch.Tensor], conf_thresh=0.001):
        """
        Decode raw outputs to boxes & scores.
        outs: list of conv outputs (B, na*(5+nc), H, W)
        returns list of decoded tensors per image: (Ndet, 6) -> (x1,y1,x2,y2, conf, cls)
        """
        batch = outs[0].shape[0]
        device = outs[0].device
        decoded_batch = []

        for b in range(batch):
            xyxy_all = []
            for i, out in enumerate(outs):
                # out_b: (na, 5+nc, H, W)
                _, _, H, W = out.shape
                na = self.na[i]
                stride = self.strides[i]
                # reshape
                out_b = out[b].view(na, self.num_outputs, H, W).permute(0, 2, 3, 1)  # (na, H, W, dim)
                # grid
                grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                gx = grid_x.view(1, H, W).float()
                gy = grid_y.view(1, H, W).float()

                # apply sigmoid to tx, ty, obj, class
                tx = torch.sigmoid(out_b[..., 0])
                ty = torch.sigmoid(out_b[..., 1])
                tw = out_b[..., 2]
                th = out_b[..., 3]
                tobj = torch.sigmoid(out_b[..., 4])
                tcls = torch.sigmoid(out_b[..., 5:])  # (na, H, W, nc)

                # compute box centers in image scale
                anchor_grid = torch.tensor(self.anchors[i], device=device).view(na, 1, 1, 2).float()
                # note: xy in pixels relative to stride
                bx = (tx + gx) * stride
                by = (ty + gy) * stride
                bw = (torch.exp(tw) * anchor_grid[..., 0])
                bh = (torch.exp(th) * anchor_grid[..., 1])

                # reshape to (num_preds, ...)
                bx = bx.view(-1)
                by = by.view(-1)
                bw = bw.view(-1)
                bh = bh.view(-1)
                tobj = tobj.view(-1)
                # Use reshape instead of view for tcls before using it
                tcls = tcls.reshape(-1, self.num_classes)  # (num_preds, nc)

                # convert to xyxy absolute coords
                x1 = bx - bw / 2
                y1 = by - bh / 2
                x2 = bx + bw / 2
                y2 = by + bh / 2

                # compute scores: objectness * class score (max)
                cls_scores, cls_ids = tcls.max(dim=1)
                scores = tobj * cls_scores

                # filter by conf_thresh
                mask = scores > conf_thresh
                if mask.sum() == 0:
                    continue
                boxes = torch.stack([x1, y1, x2, y2], dim=1)[mask]
                scores = scores[mask]
                classes = cls_ids[mask].float().unsqueeze(1)
                dets = torch.cat([boxes, scores.unsqueeze(1), classes], dim=1)  # (N,6)
                xyxy_all.append(dets)

            if len(xyxy_all) == 0:
                decoded_batch.append(torch.zeros((0, 6), device=device))
            else:
                decoded_batch.append(torch.cat(xyxy_all, dim=0))
        return decoded_batch


# -------------------------
# YOLOv12 architecture
# -------------------------
class YOLOv12Module(nn.Module):
    def __init__(self, num_classes=80, width_mult=0.25, depth_mult=0.33, anchors=None, strides=None):
        """
        A compact YOLOv12-like model:
          - backbone with C2f-like blocks (P3-P5)
          - SPPF
          - PANet (FPN + down-path)
          - Detect head with anchors & strides
        """
        super().__init__()
        self.num_classes = num_classes
        # base dimensions
        base_ch = int(64 * width_mult)
        base_depth = max(1, round(3 * depth_mult))

        # === Backbone (P3/8, P4/16, P5/32) ===
        self.stem = ConvBNAct(3, base_ch, k=3, s=1)              # /1
        self.conv1 = ConvBNAct(base_ch, base_ch * 2, k=3, s=2)    # /2
        self.c2f1 = C2f(base_ch * 2, num_blocks=base_depth)

        self.conv2 = ConvBNAct(base_ch * 2, base_ch * 4, k=3, s=2)  # /4
        self.c2f2 = C2f(base_ch * 4, num_blocks=base_depth * 2)

        self.conv3 = ConvBNAct(base_ch * 4, base_ch * 8, k=3, s=2)  # /8 (P3)
        self.c2f3 = C2f(base_ch * 8, num_blocks=base_depth * 2)

        self.conv4 = ConvBNAct(base_ch * 8, base_ch * 16, k=3, s=2) # /16 (P4)
        self.c2f4 = C2f(base_ch * 16, num_blocks=base_depth)

        self.conv5 = ConvBNAct(base_ch * 16, base_ch * 32, k=3, s=2) # /32 (P5)
        self.c2f5 = C2f(base_ch * 32, num_blocks=base_depth)
        self.sppf = SPPF(base_ch * 32, k=5)

        # === Neck (PANet) ===
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Up path
        self.reduce5 = ConvBNAct(base_ch * 32, base_ch * 16, k=1, s=1)
        self.merge4 = ConvBNAct(base_ch * 32, base_ch * 16, k=3, s=1)
        self.c2f_up1 = C2f(base_ch * 16, num_blocks=base_depth)

        self.reduce4 = ConvBNAct(base_ch * 16, base_ch * 8, k=1, s=1)
        self.merge3 = ConvBNAct(base_ch * 16, base_ch * 8, k=3, s=1)
        self.c2f_up2 = C2f(base_ch * 8, num_blocks=base_depth)

        # Down path
        self.down1 = ConvBNAct(base_ch * 8, base_ch * 16, k=3, s=2)
        self.merge_down1 = ConvBNAct(base_ch * 32, base_ch * 16, k=3, s=1)
        self.c2f_down1 = C2f(base_ch * 16, num_blocks=base_depth)

        self.down2 = ConvBNAct(base_ch * 16, base_ch * 32, k=3, s=2)
        self.merge_down2 = ConvBNAct(base_ch * 64, base_ch * 32, k=3, s=1)
        self.c2f_down2 = C2f(base_ch * 32, num_blocks=base_depth)

        # === Head ===
        # anchors & strides (defaults)
        if strides is None:
            strides = [8, 16, 32]  # P3, P4, P5
        if anchors is None:
            # anchors are given as pixel sizes relative to input
            # three anchors per stride
            anchors = [
                [(10,13),(16,30),(33,23)],    # P3 / 8
                [(30,61),(62,45),(59,119)],   # P4 /16
                [(116,90),(156,198),(373,326)] # P5 /32
            ]

        # Detect head expects features small->large [P3, P4, P5]
        head_in = [base_ch * 8, base_ch * 16, base_ch * 32]
        self.detect = DetectHead(head_in, num_classes=num_classes, anchors=anchors, strides=strides)
        self.strides = strides
        self.anchors = anchors

    def forward(self, x):
        # backbone
        x0 = self.stem(x)        # /1
        x1 = self.conv1(x0)      # /2
        x1 = self.c2f1(x1)

        x2 = self.conv2(x1)      # /4
        x2 = self.c2f2(x2)

        x3 = self.conv3(x2)      # /8 (P3)
        x3 = self.c2f3(x3)

        x4 = self.conv4(x3)      # /16 (P4)
        x4 = self.c2f4(x4)

        x5 = self.conv5(x4)      # /32 (P5)
        x5 = self.c2f5(x5)
        x_spp = self.sppf(x5)

        # === Up path ===
        # P5 -> P4
        u1 = self.reduce5(x_spp)       # reduce channels
        u1 = self.upsample(u1)         # up to /16
        cat1 = torch.cat([u1, x4], dim=1) # cat with P4
        m1 = self.merge4(cat1)
        m1 = self.c2f_up1(m1)           # P4-neck out (/16)

        # P4 -> P3
        u2 = self.reduce4(m1)
        u2 = self.upsample(u2)         # up to /8
        cat2 = torch.cat([u2, x3], dim=1) # cat with P3
        m2 = self.merge3(cat2)
        m2 = self.c2f_up2(m2)           # P3-neck out (/8)

        # === Down path ===
        # P3 -> P4
        d1 = self.down1(m2)            # down to /16
        cat_d1 = torch.cat([d1, m1], dim=1)
        d1 = self.merge_down1(cat_d1)
        d1 = self.c2f_down1(d1)        # P4-detect out (/16)

        # P4 -> P5
        d2 = self.down2(d1)            # down to /32
        cat_d2 = torch.cat([d2, x_spp], dim=1)
        d2 = self.merge_down2(cat_d2)
        d2 = self.c2f_down2(d2)        # P5-detect out (/32)

        # head outputs correspond to scales: [P3 (m2), P4 (d1), P5 (d2)]
        # These now match strides [8, 16, 32]
        outs = self.detect([m2, d1, d2])
        return outs


# -------------------------
# Loss & target assignment
# -------------------------
class YOLOLoss(nn.Module):
    def __init__(self, anchors: List[List[Tuple[int,int]]], strides: List[int], num_classes: int, img_size: int = 320,
                 cls_pw: float = 1.0, obj_pw: float = 1.0, box_loss_gain: float = 0.05):
        super().__init__()
        self.anchors = anchors
        self.strides = strides
        self.num_classes = num_classes
        self.img_size = img_size
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='sum')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='sum')
        self.cls_pw = cls_pw
        self.obj_pw = obj_pw
        self.box_loss_gain = box_loss_gain

    def build_targets(self, preds: List[torch.Tensor], targets: torch.Tensor):
        """
        Assign targets to prediction cells.
        preds: list of raw outputs from DetectHead convs (B, na*(5+nc), H, W)
        targets: tensor of shape (M,6) -> (img_idx, class, x_center, y_center, w, h) in absolute pixels
        returns list of target dicts per scale containing:
            indices (img, anchor, gy, gx), target box (xywh), class labels
        """
        device = preds[0].device
        targets = targets.clone()
        if targets.numel() == 0:
            # return empty lists
            return [dict(indices=None, tbox=None, tcls=None) for _ in preds]

        # group targets by image
        # We'll prepare per-scale lists
        t_out = []
        for i, p in enumerate(preds):
            _, _, H, W = p.shape
            stride = self.strides[i]
            na = len(self.anchors[i])
            # scale targets to feature map
            t = targets.clone()
            t_img = t[:, 0].long()
            t_cls = t[:, 1].long()
            # xy scaled to grid
            xy = t[:, 2:4] / stride  # center in grid coords
            wh = t[:, 4:6]  # absolute pixels (not scaled)
            # For anchor matching compute ratio between target w/h and anchor shapes
            anchor_shapes = torch.tensor(self.anchors[i], device=device).float()  # (na,2)
            # compute best anchor per target (by IoU in width/height)
            # convert anchor to (w,h) in pixels (they already are)
            # Expand dims to compute ratio
            wh_exp = wh.unsqueeze(1)  # (M,1,2)
            anchors_exp = anchor_shapes.unsqueeze(0)  # (1,na,2)
            ratios = (wh_exp / anchors_exp).clamp(min=1/16, max=16)
            # choose anchor with minimal max ratio distance
            max_ratios = torch.max(ratios, 2)[0]  # (M,na)
            best_anchor_idx = torch.argmin(max_ratios, dim=1)  # (M,)

            # compute grid cell indices
            gx = (xy[:, 0])
            gy = (xy[:, 1])
            gi = gx.long()
            gj = gy.long()

            # mask valid inside grid
            mask = (gi >= 0) & (gi < W) & (gj >= 0) & (gj < H)
            if mask.sum() == 0:
                # empty targets for this scale
                t_out.append(dict(indices=None, tbox=None, tcls=None))
                continue

            img_idx = t_img[mask]
            cls_idx = t_cls[mask]
            a_idx = best_anchor_idx[mask]
            gi = gi[mask]
            gj = gj[mask]
            # tbox: relative x/y within cell, and width/height in pixels
            tx = xy[mask, 0] - gi.float()
            ty = xy[mask, 1] - gj.float()
            tw = torch.log((wh[mask, 0] / anchor_shapes[a_idx][:, 0]).clamp(min=1e-6))
            th = torch.log((wh[mask, 1] / anchor_shapes[a_idx][:, 1]).clamp(min=1e-6))
            tbox = torch.stack([tx, ty, tw, th], dim=1)  # (nt,4)
            indices = torch.stack([img_idx, a_idx, gj, gi], dim=1)  # (nt,4)

            t_out.append(dict(indices=indices.long(), tbox=tbox, tcls=cls_idx.long()))
        return t_out

    def forward(self, preds: List[torch.Tensor], targets: torch.Tensor):
        """
        preds: list of outputs from DetectHead convs (B, na*(5+nc), H, W)
        targets: (M,6) -> (img_idx, class, x_center, y_center, w, h) in absolute pixels
        returns total_loss (scalar tensor) and a dict of components
        """
        device = preds[0].device
        bs = preds[0].shape[0]
        lbox = torch.tensor(0., device=device)
        lobj = torch.tensor(0., device=device)
        lcls = torch.tensor(0., device=device)

        # create targets per scale
        t_per_scale = self.build_targets(preds, targets)

        for i, p in enumerate(preds):
            B, C, H, W = p.shape
            na = len(self.anchors[i])
            stride = self.strides[i]
            p_view = p.view(B, na, self.num_classes + 5, H, W).permute(0, 1, 3, 4, 2)  # (B,na,H,W,dim)

            # objectness target map: default zeros
            tobj = torch.zeros((B, na, H, W), device=device)
            tbox = None
            tcls = None
            indices = None

            tinfo = t_per_scale[i]
            if tinfo['indices'] is not None:
                indices = tinfo['indices']  # (nt,4) img, a, gj, gi
                tbox = tinfo['tbox']        # (nt,4) tx,ty,tw,th
                tcls = tinfo['tcls']        # (nt,)
                # fill objectness at these indices = 1
                tobj[indices[:,0], indices[:,1], indices[:,2], indices[:,3]] = 1.0

                # compute box loss for the selected indices:
                # decode prediction at those indices
                pred_sel = p_view[indices[:,0], indices[:,1], indices[:,2], indices[:,3], :]
                # pred_sel: (nt, 5+nc) with tx,ty,tw,th, object, classes...
                pred_xy = torch.sigmoid(pred_sel[:, 0:2])  # tx,ty
                pred_wh = pred_sel[:, 2:4]  # tw, th (raw)
                # anchor
                anchor_shape = torch.tensor(self.anchors[i], device=device).float()
                a_idx = indices[:,1]
                anchor_for_target = anchor_shape[a_idx]  # (nt,2)
                # map to absolute coordinates in pixel space
                gx = indices[:,3].float()
                gy = indices[:,2].float()
                bx = (pred_xy[:,0] + gx) * stride
                by = (pred_xy[:,1] + gy) * stride
                bw = torch.exp(pred_wh[:,0]) * anchor_for_target[:,0]
                bh = torch.exp(pred_wh[:,1]) * anchor_for_target[:,1]
                pred_boxes_xywh = torch.stack([bx, by, bw, bh], dim=1)
                pred_boxes_xyxy = xywh2xyxy(pred_boxes_xywh)

                # build target absolute xywh
                # target tbox currently is tx,ty,tw,th relative to grid
                tx = tbox[:,0]
                ty = tbox[:,1]
                tw = tbox[:,2]
                th = tbox[:,3]
                tgt_bx = (tx + gx) * stride
                tgt_by = (ty + gy) * stride
                tgt_bw = torch.exp(tw) * anchor_for_target[:,0]
                tgt_bh = torch.exp(th) * anchor_for_target[:,1]
                tgt_boxes_xywh = torch.stack([tgt_bx, tgt_by, tgt_bw, tgt_bh], dim=1)
                tgt_boxes_xyxy = xywh2xyxy(tgt_boxes_xywh)

                # box loss (CIoU)
                lbox_scale = ciou_loss(pred_boxes_xyxy, tgt_boxes_xyxy).sum()
                lbox = lbox + lbox_scale * self.box_loss_gain

                # classification loss (for selected indices)
                if self.num_classes > 0:
                    # predicted class logits (no sigmoid yet)
                    pred_cls_logits = pred_sel[:, 5:]
                    # build target one-hot
                    tgt_cls = torch.zeros_like(pred_cls_logits, device=device)
                    tgt_cls.scatter_(1, tcls.unsqueeze(1), 1.0)
                    lcls_scale = self.bce_cls(pred_cls_logits, tgt_cls) * self.cls_pw
                    lcls = lcls + lcls_scale

            # objectness loss over whole map
            pred_obj_logits = p_view[..., 4]  # (B,na,H,W)
            lobj_scale = self.bce_obj(pred_obj_logits, tobj) * self.obj_pw
            lobj = lobj + lobj_scale

        # normalize by batch size
        total_loss = (lbox + lobj + lcls) / max(1, bs)
        return total_loss, dict(box=lbox.item(), obj=lobj.item(), cls=lcls.item())


# -------------------------
# NMS
# -------------------------
def non_max_suppression(detections: torch.Tensor, iou_thresh: float = 0.45, max_det: int = 300):
    """
    detections: (N,6) -> x1,y1,x2,y2,score,cls
    returns filtered detections
    """
    if detections.numel() == 0:
        return detections
    x1, y1, x2, y2 = detections[:,0], detections[:,1], detections[:,2], detections[:,3]
    scores = detections[:,4]
    classes = detections[:,5]

    # sort by scores
    order = scores.sort(descending=True)[1]
    keep = []
    while order.numel() > 0 and len(keep) < max_det:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        others = order[1:]
        # compute iou of i vs others
        xx1 = torch.max(x1[i], x1[others])
        yy1 = torch.max(y1[i], y1[others])
        xx2 = torch.min(x2[i], x2[others])
        yy2 = torch.min(y2[i], y2[others])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        area_i = (x2[i]-x1[i]).clamp(min=0) * (y2[i]-y1[i]).clamp(min=0)
        area_others = (x2[others]-x1[others]).clamp(min=0) * (y2[others]-y1[others]).clamp(min=0)
        union = area_i + area_others - inter
        iou = inter / (union + 1e-9)
        # filter out those with iou > thresh and same class
        mask = (iou <= iou_thresh) | (classes[others] != classes[i])
        if mask.sum() == 0:
            break
        order = others[mask]
    return detections[keep]


# -------------------------
# Top-level wrapper class for ease-of-use & compatibility
# -------------------------
class YOLOv12Model:
    """
    Wrapper for YOLOv12. Exposes a .model attribute for compatibility.
    Provides:
      - training_step(images, targets)
      - val_step(images, targets)
      - predict(images) -> post-NMS detections per image
    """

    def __init__(self, num_classes=80, device='cpu', img_size: int = 320,
                 width_mult=0.25, depth_mult=0.33):
        self.device = torch.device(device)
        self.model = YOLOv12Module(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult).to(self.device)
        self.img_size = img_size
        # instantiate loss using anchors & strides from model
        self.loss_fn = YOLOLoss(anchors=self.model.anchors, strides=self.model.strides, num_classes=num_classes, img_size=img_size)
        # convenience names
        self.nc = num_classes

    def training_step(self, images: torch.Tensor, targets: torch.Tensor, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Run forward + loss and step optimizer if provided.
        images: (B,3,H,W) float, pixel scale (0..img_size) or normalized scaled to same pixel size
        targets: (M,6) (img_idx, class, x_center, y_center, w, h) in absolute pixels
        returns: loss scalar, components dict
        """
        self.model.train()
        images = images.to(self.device)
        targets = targets.to(self.device)
        outs = self.model(images)  # list of conv outputs
        loss, components = self.loss_fn(outs, targets)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), components

    def val_step(self, images: torch.Tensor, targets: torch.Tensor):
        """
        Validation step (no optimizer step)
        returns loss and metrics (mAP calculation not included here)
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            targets = targets.to(self.device)
            outs = self.model(images)
            loss, components = self.loss_fn(outs, targets)
        return loss.item(), components

    def predict(self, images: torch.Tensor, conf_threshold: float = 0.001, iou_threshold: float = 0.45):
        """
        Predict with decoding + NMS.
        images: (B,3,H,W) pixel-scale images
        returns list of detections per image (N,6) -> x1,y1,x2,y2,score,cls
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outs = self.model(images)
            decoded = self.model.detect.inference_decode(outs, conf_thresh=conf_threshold)
            # per-image apply NMS
            results = []
            for det in decoded:
                if det.numel() == 0:
                    results.append(det.cpu())
                    continue
                det_nms = non_max_suppression(det, iou_thresh=iou_threshold)
                results.append(det_nms.cpu())
            return results

    def compute_map50(self, images: torch.Tensor, targets: torch.Tensor, conf_threshold: float = 0.001, iou_threshold: float = 0.5):
        """
        Compute mAP@0.5 for a batch. This is a simplified version for single-batch evaluation.
        """
        preds = self.predict(images, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        aps = []
        for i, pred in enumerate(preds):
            gt = targets[targets[:,0]==i]  # ground truth for this image
            if pred.numel() == 0 or gt.numel() == 0:
                continue
            pred_boxes = pred[:, :4]
            pred_cls = pred[:, 5]
            pred_scores = pred[:, 4]
            # convert gt xywh to xyxy
            gt_boxes_xywh = gt[:, 2:6]
            gt_boxes = xywh2xyxy(gt_boxes_xywh)
            gt_cls = gt[:, 1]
            # For each class in gt
            for c in gt_cls.unique():
                gt_mask = gt_cls == c
                pred_mask = pred_cls == c
                if pred_mask.sum() == 0:
                    continue
                pred_boxes_c = pred_boxes[pred_mask]
                pred_scores_c = pred_scores[pred_mask]
                gt_boxes_c = gt_boxes[gt_mask]
                # Sort preds by score
                order = pred_scores_c.argsort(descending=True)
                pred_boxes_c = pred_boxes_c[order]
                pred_scores_c = pred_scores_c[order]
                matched = torch.zeros(len(gt_boxes_c), dtype=torch.bool, device=pred.device)
                tp = torch.zeros(len(pred_boxes_c), device=pred.device)
                fp = torch.zeros(len(pred_boxes_c), device=pred.device)
                for j, pb in enumerate(pred_boxes_c):
                    ious = box_iou(pb.unsqueeze(0), gt_boxes_c).squeeze(0)
                    max_iou, max_idx = ious.max(0)
                    if max_iou > iou_threshold and not matched[max_idx]:
                        tp[j] = 1
                        matched[max_idx] = True
                    else:
                        fp[j] = 1
                # Precision-Recall curve
                tp_cum = tp.cumsum(0)
                fp_cum = fp.cumsum(0)
                recalls = tp_cum / (len(gt_boxes_c) + 1e-6)
                precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
                ap = 0.0
                for t in torch.arange(0, 1.1, 0.1, device=pred.device):
                    p = precisions[recalls >= t].max() if (recalls >= t).any() else torch.tensor(0.0, device=pred.device)
                    ap += p.item() / 11.0
                aps.append(ap)
        if len(aps) == 0:
            return 0.0
        return float(torch.tensor(aps).mean().item())

    # helpers for federated integration: convert state_dict <-> numpy arrays
    def get_parameters(self):
        """Return list of numpy arrays (like Flower NumPyClient expects)."""
        import numpy as _np
        sd = self.model.state_dict()
        return [sd[k].detach().cpu().numpy() for k in sd.keys()]

    def set_parameters(self, parameters: List):
        """Load a list of numpy arrays into model state_dict (matching key order)."""
        import numpy as _np
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if len(parameters) != len(keys):
            print(f"Warning: param list length ({len(parameters)}) != state_dict keys ({len(keys)}). Loading with strict=False.")
            strict = False
        else:
            strict = True
            
        new_sd = {}
        for k, p in zip(keys, parameters):
            if k not in sd:
                print(f"Warning: key {k} from parameters not in model state_dict.")
                continue
            if sd[k].shape != p.shape:
                print(f"Warning: shape mismatch for {k}. Model: {sd[k].shape}, Params: {p.shape}. Skipping.")
                if strict: strict=False # force non-strict if any shape mismatch
                continue # skip loading this param
            new_sd[k] = torch.tensor(_np.array(p), dtype=sd[k].dtype, device=self.device)
            
        self.model.load_state_dict(new_sd, strict=strict)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def summary(self):
        """Print a simple parameter summary"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"YOLOv12: params={total:,}, trainable={trainable:,}, num_classes={self.nc}")


# -------------------------
# Quick smoke test if executed
# -------------------------
if __name__ == "__main__":
    # quick smoke test
    m = YOLOv12Model(num_classes=3, device='cpu', img_size=320, width_mult=0.25, depth_mult=0.33)
    m.summary()
    # dummy batch
    images = torch.randn(2, 3, 320, 320)
    # dummy targets: (img_idx, class, xcenter, ycenter, w, h) absolute pixel coords
    targets = torch.tensor([
        [0, 0, 160., 160., 50., 60.],
        [1, 2, 120., 80., 30., 40.]
    ])
    
    print("\nTesting training_step...")
    loss, comp = m.training_step(images, targets)
    print("loss:", loss, "components:", comp)
    
    print("\nTesting predict...")
    preds = m.predict(images, conf_threshold=0.01)
    print("preds per image shapes:", [p.shape for p in preds])
    
    print("\nTesting mAP computation...")
    map50 = m.compute_map50(images, targets, conf_threshold=0.01)
    print("mAP@0.5:", map50)

    print("\nSmoke test passed!")