import torch
import torch.nn as nn
import torch.nn.functional as F


class yoloLoss(nn.Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 4
        self.lambda_noobj = 0.5
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 14

    # -----------------------------
    # (1) x,y,w,h → x1,y1,x2,y2 로 변환
    # -----------------------------
    def xywh_to_xyxy(self, box, index):
        """
        box: (N,4)
        index: (i,j)
        """
        i, j = index
        x = (box[:, 0] + i) * self.step
        y = (box[:, 1] + j) * self.step
        w = box[:, 2]
        h = box[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    # -----------------------------
    # (2) IoU 계산
    # -----------------------------
    def compute_iou(self, box1, box2, index):
        # xywh → xyxy
        b1 = self.xywh_to_xyxy(box1, index)
        b2 = self.xywh_to_xyxy(box2, index)

        x1 = torch.max(b1[:, 0], b2[:, 0])
        y1 = torch.max(b1[:, 1], b2[:, 1])
        x2 = torch.min(b1[:, 2], b2[:, 2])
        y2 = torch.min(b1[:, 3], b2[:, 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

        union = area1 + area2 - inter + 1e-6

        return inter / union

    # -----------------------------
    # (3) CIoU 계산
    # -----------------------------
    def Ciou_loss(self, box1, box2, index):
        """
        box1, box2: (N,4) xywh
        """
        # xywh → xyxy
        b1 = self.xywh_to_xyxy(box1, index)
        b2 = self.xywh_to_xyxy(box2, index)

        # IoU
        x1 = torch.max(b1[:, 0], b2[:, 0])
        y1 = torch.max(b1[:, 1], b2[:, 1])
        x2 = torch.min(b1[:, 2], b2[:, 2])
        y2 = torch.min(b1[:, 3], b2[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1 + area2 - inter + 1e-6
        iou = inter / union

        # 중심점 거리
        cx1 = (b1[:, 0] + b1[:, 2]) / 2
        cy1 = (b1[:, 1] + b1[:, 3]) / 2
        cx2 = (b2[:, 0] + b2[:, 2]) / 2
        cy2 = (b2[:, 1] + b2[:, 3]) / 2
        center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        # 외접 박스 diag 길이
        x_c1 = torch.min(b1[:, 0], b2[:, 0])
        y_c1 = torch.min(b1[:, 1], b2[:, 1])
        x_c2 = torch.max(b1[:, 2], b2[:, 2])
        y_c2 = torch.max(b1[:, 3], b2[:, 3])
        outer_diag = (x_c2 - x_c1) ** 2 + (y_c2 - y_c1) ** 2 + 1e-6

        # 종횡비 일관성 v, α
        w1 = b1[:, 2] - b1[:, 0]
        h1 = b1[:, 3] - b1[:, 1]
        w2 = b2[:, 2] - b2[:, 0]
        h2 = b2[:, 3] - b2[:, 1]
        v = (4 / (torch.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
        alpha = v / (1 - iou + v + 1e-6)

        ciou = iou - center_dist / outer_diag - alpha * v

        # Loss = 1 - CIoU
        return 1 - ciou

    # -----------------------------
    # (4) forward
    # -----------------------------
    def forward(self, pred, target):
        batch_size = pred.size(0)

        # bbox (10) → (B,S,S,2,5)
        target_boxes = target[:, :, :, :10].reshape(
            (-1, self.S, self.S, 2, 5))
        pred_boxes = pred[:, :, :, :10].reshape(
            (-1, self.S, self.S, 2, 5))

        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]

        # object mask
        obj_mask = (target_boxes[..., 4] > 0).bool()   # (B,S,S,2)
        sig_mask = obj_mask[..., 1]                   # (B,S,S)

        index = torch.where(sig_mask == True)

        # -------------------------
        # BBOX 책임 할당 (highest IoU)
        # -------------------------
        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x]
            tbox = target_boxes[img_i, y, x]
            ious = self.compute_iou(pbox[:, :4], tbox[:, :4], [x, y])
            _, max_i = ious.max(0)
            obj_mask[img_i, y, x, 1 - max_i] = False

        noobj_mask = ~obj_mask

        # -------------------------
        # (A) confidence loss (L2)
        # -------------------------
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

        # -------------------------
        # (B) bbox CIoU regression
        # -------------------------
        bbox_loss = 0
        for img_i, y, x in zip(*index):
            p = pred_boxes[img_i, y, x][obj_mask[img_i, y, x]]
            t = target_boxes[img_i, y, x][obj_mask[img_i, y, x]]
            bbox_loss += self.Ciou_loss(p[:, :4], t[:, :4], [x, y]).sum()

        # -------------------------
        # (C) class loss (BCE)
        # -------------------------
        class_loss = F.binary_cross_entropy(
            pred_cls[sig_mask],
            target_cls[sig_mask],
            reduction="sum"
        )

        # total
        loss = (obj_loss
                + self.lambda_noobj * noobj_loss
                + self.lambda_coord * bbox_loss
                + class_loss)

        return loss / batch_size
