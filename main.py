# -*- coding: utf-8 -*-
import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import torch.backends.cudnn as cudnn
import torch.utils.data
from calculate_error import *
from dataset.datasets_list2 import DoTADatasetSubMPM, DADADatasetSubMPM
from path import Path
from utils import *
from tadclip import *
import joblib
from tqdm import tqdm
import yaml
from datetime import datetime
import shutil

import transformers
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict

parser = argparse.ArgumentParser(description='CLIP for Traffic Anomaly Detection',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--accident_templates', nargs='+', type=str, default=['The {} vehicle collision with another {}', 'The {} vehicle out-of-control and leaving the roadway', 'the {} vehicle has an unknown accident', 'The vehicle is running normally on the road'])
parser.add_argument('--accident_prompt', nargs='+', type=str, default=['A traffic anomaly occurred in the scene', 'The traffic in this scenario is normal'])
parser.add_argument('--accident_classes', nargs='+', type=str, default=['ego', 'non-ego', 'vehicle', 'pedestrian', 'obstacle'])
# Directory setting
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--other_method', type=str, default='TDAFF_BASE')  # default='MonoCLIP'
parser.add_argument('--base_model', type=str, default='ViT-B-16', help='base model: RN50, Simba_L, ViT-B-16, ViT-B-32, RN50x64, ViT-L-14')
parser.add_argument('--trainfile_dota', type=str,
                    default="/home/haoph/Desktop/LLM/LLM/TTHF/dataset/DoTA/train_split.txt")
parser.add_argument('--testfile_dota', type=str,
                    default="/home/haoph/Desktop/LLM/LLM/TTHF/dataset/DoTA/val_split.txt")

# Dataloader setting
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr_clip', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_other', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=96, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default="DoTA")  # FIXME KITTI, NYU, DoTA
parser.add_argument('--wd', default=1e-4, type=float, help='Weight decay')
parser.add_argument("--warmup_length", type=int, default=500)
parser.add_argument("--warmup_steps", type=int, default=200)

# Model setting
parser.add_argument('--height', type=int, default=224)  # default 224(RN50), 448(RN50x64)
parser.add_argument('--width', type=int, default=224)  # default 224(RN50), 448(RN50x64)
parser.add_argument('--normal_class', type=int, default=1)
parser.add_argument('--fg', action='store_true', help='fine-grained prompts')
parser.add_argument('--general', action='store_true', help='general prompts')

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--multi_test', type=bool, default=False, help='evaluate score')

# Training setting
parser.add_argument('--train', action='store_true', help='training mode')
parser.add_argument('--exp_name', type=str, default='TDAFF_BASE_general_classifier_wo_pretrain')
parser.add_argument('--resume_model_path', type=str, default="", help='Path to the model checkpoint to resume training from')
# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default="4", help='force available gpu index')
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)
parser.add_argument('--freezen_clip', type=bool, default=False, help='node rank for distributed training')


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def to_cuda(data):
    if isinstance(data, list):
        return [to_cuda(item) for item in data]
    return data.cuda()

def validate(args, val_loader, model, dataset='KITTI'):
    val_from_save_np = False
    if val_from_save_np:
        scores = np.load("scores.npy")
        print("loading scores:", scores)
        paths = dict(log_dir="%s/%s" % (args.model_dir, args.exp_name))
        os.makedirs(paths["log_dir"], exist_ok=True)
    else:
        paths = dict(log_dir="%s/%s" % (args.model_dir, args.exp_name))
        os.makedirs(paths["log_dir"], exist_ok=True)
        ##global device
        if dataset in ['DoTA', 'DADA']:
            scores = []

        length = len(val_loader)
        # switch to evaluate mode
        model.eval()

        video_names = []
        video_scores = {}
        
        for i, batch in enumerate(val_loader):
            if args.other_method in ['TDAFF_BASE']:
                # return rgb_p, rgb_c, one_hot_label_M, one_hot_label_S,  features_p, features_c, text
                (rgb_data, rgb_data_c, _, _, features_p, features_c,  _,  video_name) = batch
                # (rgb_data, rgb_data_c, _, _, _) = batch
                rgb_data = to_cuda(rgb_data)
                rgb_data_c = to_cuda(rgb_data_c)
                features_p = to_cuda(features_p)
                features_c = to_cuda(features_c)
                with torch.no_grad():
                    if args.other_method == 'TDAFF_BASE':
                        if args.fg and args.general:
                            output_logits_m, output_logits_s = model(rgb_data, rgb_data_c, features_p, features_c, video_name, mode='eval')
                        elif args.fg:
                            output_logits_m = model(rgb_data, rgb_data_c, features_p, features_c, video_name, mode='eval')
                        else:
                            output_logits_s = model(rgb_data, rgb_data_c, features_p, features_c, video_name, mode='eval')
                    else:
                        raise ModuleNotFoundError("method not found")

            if dataset in ['DoTA', 'DADA']:
                if args.other_method in ['TDAFF_BASE']:
                    if args.other_method == 'TDAFF_BASE':
                        if args.fg and args.general:
                            output_logits_s = output_logits_s.cpu().numpy()
                            output_logits_m = output_logits_m.cpu().numpy()

                            coarse_score = 1 - output_logits_s[:, -1]
                            refine_score = 1 - output_logits_m[:, -1]
                            frame_score = (coarse_score + refine_score) / 2
                        elif args.fg:
                            output_logits_m = output_logits_m.cpu().numpy()

                            frame_score = 1 - output_logits_m[:, -1]
                        else:
                            output_logits_s = output_logits_s.cpu().numpy()

                            frame_score = 1 - output_logits_s[:, -1]
                            
            # video_name is a list containing the names of multiple videos
            for idx, name in enumerate(video_name):
                if name not in video_scores:
                    video_scores[name] = []  # Initialize an empty score list for each video name
                video_scores[name].append(frame_score[idx])  # Add the score of the current frame to the corresponding video's score list

            if i % 100 == 0:
                print('valid: {}/{}'.format(i, length))

            scores = np.append(scores, frame_score)
        #     np.save("/home/hph/Desktop/LLM/LLM/LLM/LLM/TTHF/scores.npy", scores)

        # # 保存视频名称与分数为 .npz 格式
        # np.savez('/home/hph/Desktop/LLM/LLM/LLM/LLM/TTHF/video_scores.npz', video_scores=video_scores)

    if dataset == 'DoTA':
        joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s.json" % (
            args.height, args.width)))
        gt = joblib.load(
            open(os.path.join('/home/haoph/Desktop/LLM/LLM/TTHF/dataset/DoTA', "ground_truth_demo/gt_label.json"),
                 "rb"))  # change path to DoTA dataset
        auc, sub_auc_results = compute_tad_scores(scores, gt, args, sub_test=True)
        return auc, sub_auc_results, scores
    elif dataset == 'DADA':
        joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s_dada.json" % (
            args.height, args.width)))
        gt = joblib.load(
            open(os.path.join('/data/lrq/DADA-2000', "ground_truth_demo/gt_label.json"),
                 "rb"))   # change path to DADA dataset
        auc, sub_auc_results = compute_tad_scores(scores, gt, args, sub_test=True, dataset='dada')
        return auc, sub_auc_results, scores

def train_dota(args, train_loader, val_loader, model, optimizer):

    # Generate Timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = {
        "log_dir": os.path.join(args.model_dir, timestamp, args.exp_name),
        "ckpt_dir": os.path.join(args.model_dir, timestamp, args.exp_name)
    }
    os.makedirs(paths["ckpt_dir"], exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)

    # Copy Necessary Files
    for filename in ['main.py', 'dataset/datasets_list.py', 
                     'tadclip.py', 'open_clip_local/transformer.py', 
                     'open_clip_local/model.py', 'scripts/train.sh', "utils.py"]:
        shutil.copy(filename, os.path.join(args.model_dir, timestamp))

    # Save Configuration File
    with open(os.path.join(paths["log_dir"], "clip_for_dota_cfg.yaml"), 'w') as f:
        yaml.dump(args, f)

    if args.resume_model_path:
        model, start_epoch, optimizer = load_model(args, model, optimizer)
    else:
        start_epoch = 0  # If Not Loaded, Start From Zero

    # Initialize Warm-Up Scheduler
    warmup_lambda = lambda step: min(1.0, (step + 1) / args.warmup_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Initialize ReduceLROnPlateau Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    length = len(train_loader)
    best_epoch = 0
    best_auc = 0.0
    batch_idx = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_losses = 0

        for i, batch in tqdm(enumerate(train_loader), desc="Training Epoch %d" % (epoch + 1), total=length):
            optimizer.zero_grad()
            
            # Process Batch Data
            frames, frame_c, one_hot_label_m, one_hot_label_s, features_p, features_c, _, video_name = batch 
            frames = frames.cuda() if not isinstance(frames, list) else [frame.cuda() for frame in frames]
            frame_c = frame_c.cuda() if not isinstance(frame_c, list) else [fc.cuda() for fc in frame_c]
            label_m = one_hot_label_m.cuda()
            label_s = one_hot_label_s.cuda()
            features_p = features_p.cuda() if not isinstance(features_p, list) else [fp.cuda() for fp in features_p]
            features_c = features_c.cuda()


            # Model Forward Propagation
            logits_per_frame_m, logits_per_text_m, logits_per_frame_s, logits_per_text_s = model(
                frames, frame_c, features_p, features_c, video_name, mode='train')

            losses, losses_m, losses_s = compute_losses(logits_per_frame_m, logits_per_text_m, logits_per_frame_s, logits_per_text_s, 
                                            label_m, label_s)

            total_losses += losses.item()
            losses.backward()
            optimizer.step()

            # Record Loss and Model State
            if batch_idx % 200 == 0:
                log_training_info(args, i, epoch, losses, optimizer, paths, losses_m, losses_s)

            # Learning Rate Scheduling
            if batch_idx < args.warmup_steps:
                warmup_scheduler.step()
            else:
                #  If Using ReduceLROnPlateau, Call scheduler.step(auc) After Validation
                if batch_idx > 0 and batch_idx % args.eval_every == 0:
                    auc, sub_auc_results, scores= validate(args, val_loader, model, 'DoTA')
                    model.train()
                    log_evaluation_info(args, batch_idx, epoch, auc, sub_auc_results, best_auc, paths, scores, model)

                    if auc > best_auc:
                        best_auc = auc
                        best_epoch = epoch
                        save_checkpoint({
                            'epoch': epoch,
                            'step': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_auc': best_auc
                        }, is_best=True, ckpt_dir=paths['ckpt_dir'], best_model_path=paths['ckpt_dir'], step=i, epoch=epoch)
                        save_one_model(paths['ckpt_dir'], best_epoch, max_to_save=5)

                        print('AUC result: ', auc)
                        print(f"Best model saved at epoch {best_epoch} with AUC: {best_auc}")
                    else:
                        save_checkpoint({
                            'epoch': epoch,
                            'step': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_auc': best_auc
                        }, is_best=True, ckpt_dir=paths['ckpt_dir'], best_model_path=paths['ckpt_dir'], step=i, epoch=epoch)
                        save_one_model(paths['ckpt_dir'], best_epoch, max_to_save=5)
                        print('AUC result: ', auc)
                    
                    # Adjust Learning Rate
                    scheduler.step(auc)  # Use AUC to Adjust Learning Rate

            batch_idx += 1

    return

def save_checkpoint(state, is_best, ckpt_dir, best_model_path, step, epoch):
    # Save checkpoint
    checkpoint_path = os.path.join(ckpt_dir, "epoch_%d_step_%d.pth" % (epoch + 1, step))
    torch.save(state, checkpoint_path)

    # If Current Model Is the Best Model, Save as Best Model Weights
    if is_best:
        best_model_checkpoint = os.path.join(best_model_path, "best_model.pth")
        torch.save(state, best_model_checkpoint)

def compute_losses(logits_per_frame_m, logits_per_text_m, logits_per_frame_s, logits_per_text_s, label_m, label_s):
        loss_frame_m = nn.CrossEntropyLoss()
        loss_frame_s = nn.CrossEntropyLoss()
        loss_text_s = nn.CrossEntropyLoss()
        loss_text_m = nn.CrossEntropyLoss()

        loss_img_m = loss_frame_m(logits_per_frame_m, label_m.long())
        loss_img_s = loss_frame_s(logits_per_frame_s, label_s.long())
        labels_m = label_m.t()
        labels_m_ = torch.unique(labels_m, dim=0)
        tmp_loss_m = []
        logits_per_text_m_ = logits_per_text_m.gather(0, labels_m_.unsqueeze(-1).expand(-1,
                                                                                        logits_per_text_m.shape[
                                                                                            -1]))
        for idx, tmp_class_idx in enumerate(labels_m_):
            cur_tmp_loss = [logits_per_text_m_[idx][labels_m == tmp_class_idx].mean().unsqueeze(0)]
            for cur_tmp_inner_idx in range(logits_per_text_m.shape[0]):
                if cur_tmp_inner_idx == tmp_class_idx:
                    continue
                cur_tmp_loss.append(
                    logits_per_text_m_[idx][labels_m == cur_tmp_inner_idx].mean().unsqueeze(0))
            tmp_loss_m.append(torch.cat(cur_tmp_loss))
        loss_t_m = loss_text_m(torch.stack(tmp_loss_m),
                            torch.zeros(logits_per_text_m_.shape[0]).long().to(labels_m.device))

        labels_s = label_s.t()
        labels_s_ = torch.unique(labels_s, dim=0)
        tmp_loss_s = []
        logits_per_text_s_ = logits_per_text_s.gather(0, labels_s_.unsqueeze(-1).expand(-1,
                                                                                        logits_per_text_s.shape[
                                                                                            -1]))
        for idx, tmp_class_idx in enumerate(labels_s_):
            cur_tmp_loss = [logits_per_text_s_[idx][labels_s == tmp_class_idx].mean().unsqueeze(0)]
            for cur_tmp_inner_idx in range(logits_per_text_s.shape[0]):
                if cur_tmp_inner_idx == tmp_class_idx:
                    continue
                cur_tmp_loss.append(
                    logits_per_text_s_[idx][labels_s == cur_tmp_inner_idx].mean().unsqueeze(0))
            tmp_loss_s.append(torch.cat(cur_tmp_loss))
        loss_t_s = loss_text_s(torch.stack(tmp_loss_s),
                            torch.zeros(logits_per_text_s_.shape[0]).long().to(labels_s.device))

        losses_m = loss_t_m + loss_img_m if not torch.isnan(loss_t_m).any() else loss_img_m
        losses_s = loss_t_s + loss_img_s if not torch.isnan(loss_t_s).any() else loss_img_s
        losses = (losses_m + losses_s) / 2

        return losses, losses_m, losses_s

def log_training_info(args, i, epoch, losses, optimizer, paths, losses_m, losses_s):
    print("[Step: {}/ Epoch: {}]: T_Loss: {:.4f}".format(i + 1, epoch + 1, losses))
    with open(os.path.join(paths['ckpt_dir'], "loss.txt"), 'a') as f:
        if args.other_method in ['TDAFF_BASE']:
            if args.other_method == 'TDAFF_BASE':
                if args.fg and args.general:
                    f.write(
                        "[Step: {}/ Epoch: {}]: T_Loss: {:.4f}, Loss_m: {:.4f}, Loss_s: {:.4f}, learning_rate: {:.6f}".format(
                            i + 1, epoch + 1,
                            losses,
                            losses_m, losses_s,
                            optimizer.param_groups[0]['lr']) + '\n')
                else:
                    f.write(
                        "[Step: {}/ Epoch: {}]: T_Loss: {:.4f}, learning_rate: {:.6f}".format(
                            i + 1, epoch + 1,
                            losses,
                            optimizer.param_groups[0]['lr']) + '\n')
        f.close()

def load_model(args, model, optimizer):

    if args.resume_model_path:
        if os.path.isfile(args.resume_model_path):
            print("=> loading checkpoint '{}'".format(args.resume_model_path))
            checkpoint = torch.load(args.resume_model_path, map_location='cpu')

            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            # start_epoch = checkpoint['epoch'] + 1
            start_epoch = 0
            # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume_model_path, checkpoint['epoch']))
            del checkpoint
            return model, start_epoch, optimizer  


def load_model_val(args, model):
    if args.resume_model_path:
        if os.path.isfile(args.resume_model_path):
            print("=> loading checkpoint '{}'".format(args.resume_model_path))
            checkpoint = torch.load(args.resume_model_path, map_location='cpu')

            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = 0
            
            del checkpoint
            return model, start_epoch
        
def save_model(model, optimizer, paths, best_epoch, batch_idx, epoch):
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(paths['ckpt_dir'], f"epoch_{epoch + 1}_step_{batch_idx}.pt"))
    
    save_one_model(paths['ckpt_dir'], best_epoch, max_to_save=5)

# def log_evaluation_info(args, i, epoch, auc, key, sub_auc, best_auc, paths, scores, model):
#     with open(os.path.join(paths['ckpt_dir'], "loss.txt"), 'a') as f:
#         f.write("[Step: {}/ Epoch: {}]: Eval AUC: {:.4f}".format(i + 1, epoch + 1, auc) + '\n')

#         if auc >= best_auc:
#             best_auc = auc
        
#         f.write("[Step: {}/ Epoch: {}]: Best AUC: {:.4f}".format(i + 1, epoch + 1, best_auc) + '\n')
#         f.close()

#         joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s_best.json" % (
#             args.height, args.width)))
#         only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

def log_evaluation_info(args, i, epoch, auc, sub_auc_results, best_auc, paths, scores, model):
    # Format sub_keys and sub_aucs into a compact string
    sub_info = ", ".join([f"{key}:{value:.4f}" for key, value in sub_auc_results.items()])

    with open(os.path.join(paths['ckpt_dir'], "loss.txt"), 'a') as f:
        
        f.write("[Step: {}/ Epoch: {}]: Eval AUC: {:.4f} Subclass AUCs: {}\n".format(
            i + 1, epoch + 1, auc, sub_info))

        
        if auc >= best_auc:
            best_auc = auc
        
        f.write("[Step: {}/ Epoch: {}]: Best AUC: {:.4f} sub_keys: {}\n".format(
            i + 1, epoch + 1, best_auc, sub_info))

    
    joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s_best.json" % (
        args.height, args.width)))
    
    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

def log_eval_info(args, auc, sub_auc_results, paths):
    # Format the key-value pairs of the sub_auc_results dictionary into a string
    sub_info = ", ".join([f"{key}:{value:.4f}" for key, value in sub_auc_results.items()])

    with open(os.path.join(paths, "loss.txt"), 'a') as f:
        # Write the current evaluation information, including the overall AUC and the AUC for each subclass
        f.write("[Eval AUC: {:.4f}] Subclass AUCs: {}\n".format(auc, sub_info))

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups


def main():
    args = parser.parse_args()

    ###################### Set Random Seed ###################
    set_seed(args.seed)
    print("=> No Distributed Training")
    print('=> Index of using GPU: ', args.gpu_num)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    ###################### Data Loading ###################

    if args.dataset == 'DoTA':
        if args.other_method in ['TDAFF_BASE']:
            train_set = DoTADatasetSubMPM(args, train=True)
            test_set = DoTADatasetSubMPM(args, train=False)
        else:
            raise ModuleNotFoundError("method not found")
    elif args.dataset == 'DADA':
        if args.other_method in ['TDAFF_BASE']:
            train_set = None
            test_set = DADADatasetSubMPM(args, train=False)

    print("=> Dataset: ", args.dataset)
    print("=> Data height: {}, width: {} ".format(args.height, args.width))#  [224,224]
    if train_set:
        print('=> train  samples_num: {}  '.format(len(train_set)))
    if test_set:
        print('=> test  samples_num: {}  '.format(len(test_set)))

    train_sampler = None
    test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=10, pin_memory=True, sampler=train_sampler)

    if train_set:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=False,
            num_workers=10, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = None
    if test_set:
        val_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=10, pin_memory=True, sampler=test_sampler)
    else:
        val_loader = None

    cudnn.benchmark = True 

    ###################### Setting Network Part ###################

    print("=> creating model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = "pretrain_models/vicuna-7b-v1.5",
        cache_dir= None,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    clip_model, _, _ = clip.create_model_and_transforms(args.base_model, pretrained='openai', jit=False,
                                                        cache_dir='./pretrain_models')
    Model = TDAFF_BASE(args, clip_model, tokenizer)


    ###################### Freeze Network Parameters ###################

    if args.freezen_clip:
        with open('parameter_status.txt', 'w') as f:
            for name, param in Model.named_parameters():
                if ('visual' in name and 'side' not in name and 'ln_post' not in name and 'visual.proj' not in name) or 'logit_scale' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                f.write(f'{name} {param.requires_grad}\n')

    ###################### Statistics of Training or Frozen Parameters ###################
    num_params_t = 0
    num_params_f = 0
    
    num_params_t = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    num_params_f = sum(p.numel() for p in Model.parameters() if not p.requires_grad)
    print("===============================================")
    print("Training parameters: {}".format(num_params_t))
    print("Freezen parameters: {}".format(num_params_f))
    print("===============================================")

    Model = Model.cuda()
    ###################### Evaluate Network Configuration #################################

    if args.evaluate is True:
        ###################### setting model list #################################
        if args.multi_test is True:
            print("=> all of model tested")
            models_list_dir = Path(args.models_list_dir)
            models_list = sorted(models_list_dir.files('*.pkl'))
        else:
            print("=> just one model tested")
            models_list = [args.model_dir]
        test_model = Model

        print("Model Initialized")

        test_len = len(models_list)
        print("=> Length of model list: ", test_len)
        if args.resume_model_path:
            test_model, start_epoch = load_model_val(args, test_model)
        test_model.eval()
        paths = "/home/hph/Desktop/LLM/LLM/LLM/LLM/TTHF"
        for i in range(test_len):
            if args.dataset == 'DoTA':
                auc, sub_auc_results, scores = validate(args, val_loader, test_model, 'DoTA')
            elif args.dataset == 'DADA':
                auc, sub_auc_results, scores = validate(args, val_loader, test_model, 'DADA')
                log_eval_info(args, auc, sub_auc_results, paths)
            print(' * model: {}'.format(models_list[i]))
            print("")
            print(' AUC result: ', auc)
            print("")
        print(args.dataset, " valdiation finish")

    else:
        print("Model Initialized")
        train_model = Model
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr_clip, weight_decay=args.wd)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
        print("=> Training")
        if args.dataset == 'DoTA':
            train_dota(args, train_loader, val_loader, train_model, optimizer)
        print("")

if __name__ == "__main__":
    main()



