import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class yoloLoss(nn.Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 7        # 작은 물체 penalty 강화
        self.lambda_noobj = 0.2      # background loss 감소
        self.lambda_cls = 1.5        # class loss weight 증가
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / self.S

    # -----------------------------
    # 벡터화된 xywh → xyxy 변환
    # -----------------------------
    def get_absolute_coords(self, boxes, device):
        """Grid offset를 절대 좌표로 변환 (벡터화)"""
        B, S, _, NumBox, _ = boxes.shape
        y_grid, x_grid = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device), indexing='ij')
        x_grid = x_grid.view(1, S, S, 1).expand(B, S, S, NumBox)
        y_grid = y_grid.view(1, S, S, 1).expand(B, S, S, NumBox)

        txty = boxes[..., :2]
        twth = boxes[..., 2:4]
        
        cx = (txty[..., 0] + x_grid) * self.step
        cy = (txty[..., 1] + y_grid) * self.step
        
        w = twth[..., 0]
        h = twth[..., 1]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    # -----------------------------
    # 벡터화된 IoU 계산 (매칭용)
    # -----------------------------
    def compute_iou_vectorized(self, box1, box2):
        """벡터화된 IoU 계산"""
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = area1 + area2 - inter + 1e-6

        return inter / union

    # -----------------------------
    # 벡터화된 CIoU loss
    # -----------------------------
    def ciou_loss_vectorized(self, box1, box2):
        """벡터화된 CIoU loss 계산"""
        # IoU
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = area1 + area2 - inter + 1e-6

        iou = inter / union

        # 중심점 거리
        cx1 = (box1[..., 0] + box1[..., 2]) / 2
        cy1 = (box1[..., 1] + box1[..., 3]) / 2
        cx2 = (box2[..., 0] + box2[..., 2]) / 2
        cy2 = (box2[..., 1] + box2[..., 3]) / 2
        center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # 외접 박스
        x_c1 = torch.min(box1[..., 0], box2[..., 0])
        y_c1 = torch.min(box1[..., 1], box2[..., 1])
        x_c2 = torch.max(box1[..., 2], box2[..., 2])
        y_c2 = torch.max(box1[..., 3], box2[..., 3])
        outer_diag = (x_c2 - x_c1) ** 2 + (y_c2 - y_c1) ** 2 + 1e-6

        # 종횡비 일관성 v, α
        w1 = (box1[..., 2] - box1[..., 0]).clamp(min=1e-6)
        h1 = (box1[..., 3] - box1[..., 1]).clamp(min=1e-6)
        w2 = (box2[..., 2] - box2[..., 0]).clamp(min=1e-6)
        h2 = (box2[..., 3] - box2[..., 1]).clamp(min=1e-6)

        v = (4 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
        alpha = v / (1 - iou + v + 1e-6)

        ciou = iou - center_dist / outer_diag - alpha * v

        # 작은 물체 가중치
        scale = 2 - (w2 * h2)
        scale = scale.clamp(min=1.0)

        return (1 - ciou) * scale

    # -----------------------------
    # forward (벡터화 최적화)
    # -----------------------------
    def forward(self, pred, target):
        batch_size = pred.size(0)
        device = pred.device

        target_boxes = target[:, :, :, :10].reshape(batch_size, self.S, self.S, 2, 5)
        pred_boxes = pred[:, :, :, :10].reshape(batch_size, self.S, self.S, 2, 5)

        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]

        obj_mask = (target_boxes[..., 4] > 0).bool()
        sig_mask = obj_mask.any(dim=-1)  # 각 셀에 객체가 있는지

        # 절대 좌표로 변환 (벡터화)
        pred_abs = self.get_absolute_coords(pred_boxes, device)
        target_abs = self.get_absolute_coords(target_boxes, device)

        # 책임 할당 (벡터화된 매칭)
        # 각 셀에서 두 박스 중 IoU가 높은 것을 선택
        batch_idx, y_idx, x_idx = torch.where(sig_mask)
        
        for b, y, x in zip(batch_idx, y_idx, x_idx):
            p_box = pred_abs[b, y, x]  # (2, 4)
            t_box = target_abs[b, y, x, 0].unsqueeze(0)  # (1, 4)
            
            # 각 예측 박스와 타겟 박스의 IoU 계산
            ious = self.compute_iou_vectorized(p_box.unsqueeze(0), t_box)  # (2, 1)
            ious = ious.squeeze(-1)  # (2,)
            _, max_i = ious.max(0)
            
            # 선택되지 않은 박스는 마스크 제거
            obj_mask[b, y, x, 1 - max_i] = False

        noobj_mask = ~obj_mask

        # Confidence loss
        noobj_loss = F.mse_loss(
            pred_boxes[noobj_mask][:, 4],
            target_boxes[noobj_mask][:, 4],
            reduction="sum"
        )
        obj_loss = F.mse_loss(
            pred_boxes[obj_mask][:, 4],
            target_boxes[obj_mask][:, 4],
            reduction="sum"
        )

        # BBox loss (벡터화)
        pred_coord = pred_abs[obj_mask]
        target_coord = target_abs[obj_mask]
        
        ciou_loss = self.ciou_loss_vectorized(pred_coord, target_coord)
        bbox_loss = ciou_loss.sum()

        # Class loss (weighted)
        sig_mask = obj_mask.any(dim=-1)
        class_loss = F.binary_cross_entropy(
            pred_cls[sig_mask],
            target_cls[sig_mask],
            reduction="sum"
        ) * self.lambda_cls

        loss = (
            obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_coord * bbox_loss
            + class_loss
        )

        return loss / batch_size
