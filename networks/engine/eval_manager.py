import os
import importlib
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms 
import numpy as np
from dataloaders.datasets import YOUTUBE_VOS_Test, DAVIS_Test, EVAL_TEST
import dataloaders.custom_transforms as tr
from networks.deeplab.deeplab import DeepLab
from utils.meters import AverageMeter
from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder

class Evaluator(object):
    def __init__(self, cfg):
        self.gpu = cfg.TEST_GPU_ID
        self.cfg = cfg
        self.print_log(cfg.__dict__)
        print("Use GPU {} for evaluating".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        self.print_log('Build backbone.')
        self.feature_extracter = DeepLab(
            backbone=cfg.MODEL_BACKBONE,
            freeze_bn=cfg.MODEL_FREEZE_BN).cuda(self.gpu)

        self.print_log('Build VOS model.')
        CFBI = importlib.import_module(cfg.MODEL_MODULE)
        self.model = CFBI.get_module()(
            cfg,
            self.feature_extracter).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg
        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return
        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE, cfg.TEST_FLIP, cfg.TEST_MULTISCALE), 
            tr.MultiToTensor()])
        
        eval_name = '{}_{}_ckpt_{}'.format(cfg.TEST_DATASET, cfg.EXP_NAME, self.ckpt)
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms'

        if cfg.TEST_DATASET == 'youtubevos':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            self.dataset = YOUTUBE_VOS_Test(
                root=cfg.DIR_YTB_EVAL, 
                transform=eval_transforms,  
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=cfg.TEST_DATASET_SPLIT, 
                root=cfg.DIR_DAVIS, 
                year=2017, 
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION, 
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=cfg.TEST_DATASET_SPLIT, 
                root=cfg.DIR_DAVIS, 
                year=2016, 
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION, 
                result_root=self.result_root)
        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)
        else:
            print('Unknown dataset!')
            exit()

        print('Eval {} on {}:'.format(cfg.EXP_NAME, cfg.TEST_DATASET))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)
        self.print_log('Done!')

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0 
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        for seq_idx, seq_dataset in enumerate(self.dataset):
            video_num += 1
            seq_name = seq_dataset.seq_name
            print('Prcessing Seq {} [{}/{}]:'.format(seq_name, video_num, total_video_num))
            torch.cuda.empty_cache()

            seq_dataloader=DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=cfg.TEST_WORKERS, pin_memory=True)
            
            seq_total_time = 0
            seq_total_frame = 0
            ref_embeddings = []
            ref_masks = []
            prev_embedding = []
            prev_mask = []
            with torch.no_grad():
                for frame_idx, samples in enumerate(seq_dataloader):
                    time_start = time.time()
                    all_preds = []
                    join_label = None
                    for aug_idx in range(len(samples)):
                        if len(ref_embeddings) <= aug_idx:
                            ref_embeddings.append([])
                            ref_masks.append([])
                            prev_embedding.append(None)
                            prev_mask.append(None)

                        sample = samples[aug_idx]
                        ref_emb = ref_embeddings[aug_idx]
                        ref_m = ref_masks[aug_idx]
                        prev_emb = prev_embedding[aug_idx]
                        prev_m = prev_mask[aug_idx]

                        current_img = sample['current_img']
                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(self.gpu)
                        else:
                            current_label = None

                        obj_num = sample['meta']['obj_num']
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        current_img = current_img.cuda(self.gpu)
                        obj_num = obj_num.cuda(self.gpu)
                        bs, _, h, w = current_img.size()

                        all_pred, current_embedding = self.model.forward_for_eval(ref_emb, ref_m, prev_emb, prev_m, current_img, gt_ids=obj_num, pred_size=[ori_height,ori_width])

                        if frame_idx == 0:
                            if current_label is None:
                                print("No first frame label in Seq {}.".format(seq_name))
                            ref_embeddings[aug_idx].append(current_embedding)
                            ref_masks[aug_idx].append(current_label)
                            
                            prev_embedding[aug_idx] = current_embedding
                            prev_mask[aug_idx] = current_label
                        else:
                            if sample['meta']['flip']:
                                all_pred = flip_tensor(all_pred, 3)
                            #  In YouTube-VOS, not all the objects appear in the first frame for the first time. Thus, we
                            #  have to introduce new labels for new objects, if necessary.
                            if not sample['meta']['flip'] and not(current_label is None) and join_label is None:
                                join_label = current_label
                            all_preds.append(all_pred)
                            if current_label is not None:
                                ref_embeddings[aug_idx].append(current_embedding)
                            prev_embedding[aug_idx] = current_embedding

                    if frame_idx > 0:
                        all_preds = torch.cat(all_preds, dim=0)
                        all_preds = torch.mean(all_preds, dim=0)
                        pred_label = torch.argmax(all_preds, dim=0)
                        if join_label is not None:
                            join_label = join_label.squeeze(0).squeeze(0)
                            keep = (join_label == 0).long()
                            pred_label = pred_label * keep + join_label * (1 - keep)
                            pred_label = pred_label
                        current_label = pred_label.view(1, 1, ori_height, ori_width)
                        flip_pred_label = flip_tensor(pred_label, 1)
                        flip_current_label = flip_pred_label.view(1, 1, ori_height, ori_width)

                        for aug_idx in range(len(samples)):
                            if join_label is not None:
                                if samples[aug_idx]['meta']['flip']:
                                    ref_masks[aug_idx].append(flip_current_label)
                                else:
                                    ref_masks[aug_idx].append(current_label)
                            if samples[aug_idx]['meta']['flip']:
                                prev_mask[aug_idx] = flip_current_label
                            else:
                                prev_mask[aug_idx] = current_label

                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        seq_total_frame += 1
                        obj_num = obj_num[0].item()
                        print('Frame: {}, Obj Num: {}, Time: {}'.format(imgname[0], obj_num, one_frametime))
                        # Save result
                        save_mask(pred_label, os.path.join(self.result_root, seq_name, imgname[0].split('.')[0]+'.png'))
                    else:
                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        print('Ref Frame: {}, Time: {}'.format(imgname[0], one_frametime))

                del(ref_embeddings)
                del(ref_masks)
                del(prev_embedding)
                del(prev_mask)
                del(seq_dataset)
                del(seq_dataloader)

            seq_avg_time_per_frame = seq_total_time / seq_total_frame
            total_time += seq_total_time
            total_frame += seq_total_frame
            total_avg_time_per_frame = total_time / total_frame
            total_sfps += seq_avg_time_per_frame
            avg_sfps = total_sfps / (seq_idx + 1)
            print("Seq {} FPS: {}, Total FPS: {}, FPS per Seq: {}".format(seq_name, 1./seq_avg_time_per_frame, 1./total_avg_time_per_frame, 1./avg_sfps))

        zip_folder(self.source_folder, self.zip_dir)
        self.print_log('Save result to {}.'.format(self.zip_dir))
        

    def print_log(self, string):
        print(string)




