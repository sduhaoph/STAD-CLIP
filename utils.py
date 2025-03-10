from __future__ import division
import torch
import os
import glob

def save_one_model(ckpt_dir, best_epoch, max_to_save=5):
    # Find all checkpoint files
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "epoch_*_step_*.pth"))
    
    # If the number of checkpoint files exceeds max_to_save, delete the oldest ones
    if len(ckpt_files) > max_to_save:
        # Sort the files by modification time, with the earliest files placed first
        ckpt_files.sort(key=os.path.getmtime)
        
        # Number of files to delete
        num_to_delete = len(ckpt_files) - max_to_save
        
        # Delete the earliest num_to_delete files, skipping the best_epoch file
        for file in ckpt_files:
            if num_to_delete <= 0:
                break
            if f"epoch_{best_epoch}_" not in file:
                os.remove(file)
                print(f"Deleted old checkpoint: {file}")
                num_to_delete -= 1

def only_model_saver(model_state_dict, model_path):
    state_dict = {}
    state_dict["model_state_dict"] = model_state_dict
    torch.save(state_dict, model_path)
    print('models {} save successfully!'.format(model_path))



from itertools import repeat
import collections.abc

import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import numpy as np
from PIL import Image

from torchvision import models, transforms




# class VGG16(nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         # VGG = models.vgg16(pretrained=True)
#         VGG = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         self.feature = VGG.features
#         self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
#         pretrained_dict = VGG.state_dict()
#         model_dict = self.classifier.state_dict()
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict)
#         self.classifier.load_state_dict(model_dict)
#         self.dim_feat = 4096

#     def forward(self, x):
#         output = self.feature(x)
#         output = output.view(output.size(0), -1)
#         output = self.classifier(output)
#         return output
    
# def init_feature_extractor(backbone='vgg16', device=torch.device('cuda')):
#     feat_extractor = None
#     if backbone == 'vgg16':
#         feat_extractor = VGG16()
#         feat_extractor = feat_extractor.to(device=device)
#         feat_extractor.eval()
#     else:
#         raise NotImplementedError
#     return feat_extractor

# def bbox_sampling(bbox_result, nbox=19, imsize=None, topN=5):
#     """
#     从 bbox_result 中提取目标边界框，选择置信度最高的目标框，并对其进行处理和采样
    
#     参数:
#         bbox_result (DetDataSample): 包含检测结果的 DetDataSample 对象。
#         nbox (int): 目标框的总数量。
#         imsize (tuple): 图像的尺寸，(height, width)。
#         topN (int): 选择的目标框数量。
    
#     返回:
#         np.ndarray: 选择的目标框信息，形状为 [n, 6]，每一行是 [x1, y1, x2, y2, score, label]
#     """
#     # 从 DetDataSample 对象中提取目标框信息
#     pred_instances = bbox_result.pred_instances
#     bboxes = pred_instances.bboxes.cpu().numpy()  # n x 4
#     scores = pred_instances.scores.cpu().numpy()  # n
#     labels = pred_instances.labels.cpu().numpy()  # n
    
#     if len(bboxes) == 0:
#         # 如果没有检测到任何目标框，返回默认的目标框
#         return np.array([[0, 0, imsize[1]-1, imsize[0]-1, 1.0, 0]], dtype=float)

#     # 修正目标框和置信度
#     new_boxes = []
#     for box, score, label in zip(bboxes, scores, labels):
#         x1 = min(max(0, int(box[0])), imsize[1])
#         y1 = min(max(0, int(box[1])), imsize[0])
#         x2 = min(max(x1 + 1, int(box[2])), imsize[1])
#         y2 = min(max(y1 + 1, int(box[3])), imsize[0])
#         if (y2 - y1 + 1 > 2) and (x2 - x1 + 1 > 2):
#             new_boxes.append([x1, y1, x2, y2, score, label])
    
#     new_boxes = np.array(new_boxes, dtype=float)  # 转换为 numpy 数组，类型为 float

#     if len(new_boxes) == 0:  # 如果没有目标框
#         new_boxes = np.array([[0, 0, imsize[1]-1, imsize[0]-1, 1.0, 0]], dtype=float)
    
#     # 采样目标框
#     n_candidate = min(topN, len(new_boxes))
#     if len(new_boxes) <= nbox - n_candidate:
#         indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=True)
#         sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
#     elif len(new_boxes) > nbox - n_candidate and len(new_boxes) <= nbox:
#         indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=False)
#         sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
#     else:
#         sampled_boxes = new_boxes[:nbox]
    
#     # 将数据类型转换为 int
#     sampled_boxes = np.array(sampled_boxes, dtype=int)  # Convert to integer
#     return sampled_boxes

# def bbox_to_imroi(transform, bboxes, image):
#     imroi_data = []
#     for bbox in bboxes:
#         imroi = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
#         imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
#         imroi_data.append(imroi)
#     imroi_data = torch.stack(imroi_data)
#     return imroi_data
class ConvPool1d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvPool1d, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=574, stride=1, padding=0)
    
    def forward(self, x):
        # x.shape = [bs, 574, 1024]
        x = x.permute(0, 2, 1)  # [bs, 1024, 574]
        x = self.conv(x)  # [bs, 1024, 1]
        return x.squeeze(-1)  # [bs, 1024]

class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        # x.shape = [bs, 574, 1024]
        return torch.mean(x, dim=1)  # [bs, 1024]