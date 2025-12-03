import os
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data

import cv2


class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, root, file_names, train, transform):
        print('DATA INITIALIZATION')

        self.root_images = os.path.join(root, 'Images')
        self.root_labels = os.path.join(root, 'Labels')
        self.train = train
        self.transform = transform
        self.f_names = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

        for line in file_names:
            line = line.rstrip()
            with open(f"{self.root_labels}/{line}.txt") as f:
                objects = f.readlines()
                self.f_names.append(line + '.jpg')
                box = []
                label = []
                for obj in objects:
                    c, x1, y1, x2, y2 = map(float, obj.rstrip().split())
                    box.append([x1, y1, x2, y2])
                    label.append(int(c) + 1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.boxes)

    # ------------------------------------------------------
    #                    Mosaic Augmentation
    # ------------------------------------------------------
    def load_image_and_label(self, index):
        """ 기본 이미지 + box 로딩 """
        f_name = self.f_names[index]
        img = cv2.imread(os.path.join(self.root_images, f_name))
        boxes = self.boxes[index].clone()
        labels = self.labels[index].clone()
        return img, boxes, labels

    def mosaic_augmentation(self, index):
        """ 4장의 이미지를 모자이크로 합성 """
        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]
        imgs, bboxes, blabels = [], [], []

        for idx in indices:
            img, box, label = self.load_image_and_label(idx)
            imgs.append(img)
            bboxes.append(box)
            blabels.append(label)

        final_img = np.full((self.image_size * 2, self.image_size * 2, 3), 128, dtype=np.uint8)
        final_boxes = []
        final_labels = []

        # 분할 위치
        xc = random.randint(int(self.image_size * 0.5), int(self.image_size * 1.5))
        yc = random.randint(int(self.image_size * 0.5), int(self.image_size * 1.5))

        positions = [
            (0, 0, xc, yc),                                        # top-left
            (xc, 0, self.image_size * 2, yc),                      # top-right
            (0, yc, xc, self.image_size * 2),                      # bottom-left
            (xc, yc, self.image_size * 2, self.image_size * 2)     # bottom-right
        ]

        for i, (img, box, label) in enumerate(zip(imgs, bboxes, blabels)):
            h, w, _ = img.shape
            # resize
            img_resized = cv2.resize(img, (self.image_size, self.image_size))
            b_scale = torch.tensor([w, h, w, h]).float()
            box = box / b_scale
            box = box * self.image_size  # 다시 pixel 좌표로

            # mosaic 영역
            x1, y1, x2, y2 = positions[i]

            # 넣을 이미지 좌표
            img_x1 = x1
            img_y1 = y1
            img_x2 = x1 + self.image_size
            img_y2 = y1 + self.image_size

            final_img[img_y1:img_y2, img_x1:img_x2] = img_resized

            # box 이동
            shift = torch.tensor([img_x1, img_y1, img_x1, img_y1])
            moved_box = box + shift

            final_boxes.append(moved_box)
            final_labels.append(label)

        final_boxes = torch.cat(final_boxes, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        # 2배 크기 → 448×448로 리사이즈
        final_img = cv2.resize(final_img, (self.image_size, self.image_size))
        scale = 2.0
        final_boxes /= scale

        return final_img, final_boxes, final_labels

    # ------------------------------------------------------
    #      Dataset __getitem__
    # ------------------------------------------------------
    def __getitem__(self, idx):

        # --------------------------------------------------
        # Mosaic 사용 (p=0.5 정도)
        # --------------------------------------------------
        if self.train and random.random() < 0.5:
            img, boxes, labels = self.mosaic_augmentation(idx)
        else:
            img, boxes, labels = self.load_image_and_label(idx)

            # 기존 증강
            if self.train:
                img, boxes = self.random_flip(img, boxes)
                img, boxes = self.randomScale(img, boxes)
                img = self.randomBlur(img)
                img = self.RandomBrightness(img)
                img = self.RandomHue(img)
                img = self.RandomSaturation(img)
                img, boxes, labels = self.randomShift(img, boxes, labels)
                img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.tensor([w, h, w, h]).expand_as(boxes)

        img = self.BGR2RGB(img)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))

        target = self.encoder(boxes, labels)

        for t in self.transform:
            img = t(img)

        return img, target

    # ------------------------------------------------------
    # 그대로 두는 부분들 (encoder + augmentation)
    # ------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1

            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size

            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy

        return target

    # --------------------- utility augmentations ------------------------
    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after = np.zeros((height, width, c), dtype=bgr.dtype)
            after[:, :, :] = (104, 117, 123)

            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            x_shift = int(shift_x)
            y_shift = int(shift_y)

            # image shift
            if shift_x >= 0 and shift_y >= 0:
                after[y_shift:, x_shift:, :] = bgr[:height - y_shift, :width - x_shift]
            elif shift_x >= 0 and shift_y < 0:
                after[:height + y_shift, x_shift:, :] = bgr[-y_shift:, :width - x_shift]
            elif shift_x < 0 and shift_y >= 0:
                after[y_shift:, :width + x_shift, :] = bgr[:height - y_shift, -x_shift:]
            else:
                after[:height + y_shift, :width + x_shift, :] = bgr[-y_shift:, -x_shift:]

            shift_xy = torch.FloatTensor([[x_shift, y_shift]]).expand_as(center)
            center = center + shift_xy

            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels

            box_shift = torch.FloatTensor([[x_shift, y_shift, x_shift, y_shift]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]

            return after, boxes_in, labels_in

        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr_resized = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr_resized, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape

            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)

            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)

            if len(boxes_in) == 0:
                return bgr, boxes, labels

            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)
            boxes_in = boxes_in - box_shift

            boxes_in[:, 0] = boxes_in[:, 0].clamp(0, w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp(0, w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp(0, h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp(0, h)

            img_crop = bgr[y:y + h, x:x + w]
            labels_in = labels[mask.view(-1)]

            return img_crop, boxes_in, labels_in

        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        return bgr - mean

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = np.clip(im, 0, 255).astype(np.uint8)
        return im