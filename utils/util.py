import cv2
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sys
sys.path.append(r'/home/user/workspace/yolov1')
from nets.nn import resnet50
import torchvision.transforms as transforms
#from torchvision.ops import nms

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)}

def decoder(prediction):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_num = 14
    cell_size = 1. / grid_num

    prediction = prediction.data.squeeze()  # (14,14,30)

    contain = torch.stack([
        prediction[:,:,4], 
        prediction[:,:,9]
    ], dim=2)  # (14,14,2)

    # confidence threshold
    mask1 = contain > 0.1 

    # per-cell local max
    max_idx = torch.argmax(contain, dim=2, keepdim=True)
    mask2 = torch.zeros_like(contain, dtype=torch.bool)
    mask2.scatter_(2, max_idx, True)

    mask = (mask1 & mask2)

    boxes = []
    cls_indexes = []
    confidences = []

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i,j,b]:
                    # cx, cy, w, h
                    cx = (prediction[i,j,b*5] + j) * cell_size
                    cy = (prediction[i,j,b*5+1] + i) * cell_size
                    w  = prediction[i,j,b*5+2]
                    h  = prediction[i,j,b*5+3]

                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2

                    box = torch.tensor([x1,y1,x2,y2]).to(device)

                    # class score
                    max_prob, cls_index = torch.max(prediction[i,j,10:], 0)
                    conf = prediction[i,j,b*5+4] * max_prob

                    if conf > 0.1:
                        boxes.append(box.unsqueeze(0))
                        cls_indexes.append(cls_index.unsqueeze(0))
                        confidences.append(conf.unsqueeze(0))

    if len(boxes)==0:
        return torch.zeros((1,4)), torch.zeros(1), torch.zeros(1)

    boxes = torch.cat(boxes).to(device)
    confidences = torch.cat(confidences).to(device)
    cls_indexes = torch.cat(cls_indexes).to(device)

    # NMS
    keep = nms(boxes, confidences, threshold=0.45)

    return boxes[keep], cls_indexes[keep], confidences[keep]


def nms(b_boxes, scores, threshold=0.5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = b_boxes[:, 0].to(device)
    y1 = b_boxes[:, 1].to(device)
    x2 = b_boxes[:, 2].to(device)
    y2 = b_boxes[:, 3].to(device)
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:

        i = order.item() if (order.numel() == 1) else order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection

        over = intersection / union
        ids = (over <= threshold).nonzero().squeeze()
        # ids = torch.nonzero(ids, as_tuple=False).squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)

def predict(model, img_name, root_path=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    img = cv2.imread(root_path + img_name)
    h, w, _ = img.shape
    img = cv2.resize(img, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = img[None, :, :, :]
    img = img.to(device)

    prediction = model(img).to(device)  # 1x14x14x30
    boxes, cls_indexes, confidences = decoder(prediction)

    #정규화되고 시작점과 끝점으로 표현된 bbox를 original image 상에서 크기로 변환함
    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexes[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        conf = confidences[i]
        conf = float(conf)
        results.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], img_name, conf])
    return results


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)

    print('LOADING MODEL...')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('./weights/yolov1_0010.pth')['state_dict'])
    model.eval()
    
    with torch.no_grad():
        image_name = './assets/person.jpg'
        image = cv2.imread(image_name)
        print('\nPREDICTING...')
        result = predict(model, image_name)

    for x1y1, x2y2, class_name, _, prob in result:
        color = COLORS[class_name]
        cv2.rectangle(image, x1y1, x2y2, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        p1 = (x1y1[0], x1y1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv2.imwrite('./result.jpg', image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(image)
    plt.show()
