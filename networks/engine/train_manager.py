import os
import importlib
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms 
import numpy as np
from dataset.VEHI_COLOR import VECOLOR
import dataset.custom_transforms as tr
from networks.res2net.res2net_v1b import res2net101_v1b
from utils.meters import AverageMeter
from utils.checkpoint import load_network_and_optimizer, load_network, save_network
from utils.learning import adjust_learning_rate, get_trainable_params
from utils.metric import pytorch_acc


class Trainer(object):
    def __init__(self , rank, cfg):
        self.gpu = rank + cfg.DIST_START_GPU
        self.rank = rank
        self.cfg = cfg
        self.print_log(cfg.__dict__)
        print("Use GPU {} for training".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        self.print_log('Building model.')
        self.model = res2net101_v1b(
                            pretrained=cfg.PRETRAIN, 
                            num_classes=cfg.MODEL_NUM_CLASSES).cuda(self.gpu)

        if cfg.DIST_ENABLE:
            dist.init_process_group(
                backend=cfg.DIST_BACKEND, 
                init_method=cfg.DIST_URL,
                world_size=cfg.TRAIN_GPUS, 
                rank=rank, 
                timeout=datetime.timedelta(seconds=100))
            self.dist_model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.gpu],
                find_unused_parameters=False)
        else:
            self.dist_model = self.model

        self.print_log('Building optimizer.')
        trainable_params = get_trainable_params(
            model=self.dist_model, 
            base_lr=cfg.TRAIN_LR, 
            weight_decay=cfg.TRAIN_WEIGHT_DECAY, 
            beta_wd=cfg.MODEL_GCT_BETA_WD)

        self.optimizer = optim.SGD(
            trainable_params,
            lr=cfg.TRAIN_LR, 
            momentum=cfg.TRAIN_MOMENTUM, 
            nesterov=True)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.prepare_dataset()
        self.process_pretrained_model()

    def process_pretrained_model(self):
        cfg = self.cfg

        self.step = cfg.TRAIN_START_STEP
        self.epoch = 0

        if cfg.TRAIN_AUTO_RESUME:
            ckpts = os.listdir(cfg.DIR_CKPT)
            if len(ckpts) > 0:
                ckpts = list(map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts))
                ckpt = np.sort(ckpts)[-1]
                cfg.TRAIN_RESUME = True
                cfg.TRAIN_RESUME_CKPT = ckpt
                cfg.TRAIN_RESUME_STEP = ckpt + 1
            else:
                cfg.TRAIN_RESUME = False

        if cfg.TRAIN_RESUME:
            resume_ckpt = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % (cfg.TRAIN_RESUME_CKPT))

            self.model, self.optimizer, removed_dict = load_network_and_optimizer(self.model, self.optimizer, resume_ckpt, self.gpu)

            if len(removed_dict) > 0:
                self.print_log('Remove {} from checkpoint.'.format(removed_dict))

            self.step = cfg.TRAIN_RESUME_STEP
            if cfg.TRAIN_TOTAL_STEPS <= self.step:
                self.print_log("Your training has finished!")
                exit()
            self.epoch = int(np.ceil(self.step / len(self.trainloader)))

            self.print_log('Resume from step {}'.format(self.step))

        elif cfg.PRETRAIN:
            print("This version state dict has been loaded...")

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        composed_transforms = transforms.Compose([
            tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR, cfg.DATA_MAX_SCALE_FACTOR, cfg.DATA_SHORT_EDGE_LEN),
            tr.RandomCrop(cfg.DATA_RANDOMCROP), 
            tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
            tr.Resize(cfg.DATA_RANDOMCROP),
            tr.ToTensor()])
        
        train_datasets = []
        if 'vecolor' in cfg.DATASETS:
            train_vecolor_dataset = VECOLOR(
                    root=cfg.DIR_VECOLOR,
                    phase='train',
                    transform=composed_transforms
            )
            train_datasets.append(train_vecolor_dataset)

        val_transforms = transforms.Compose([
            tr.Resize(cfg.DATA_RANDOMCROP),
            tr.ToTensor()])
        val_vecolor_dataset = VECOLOR(
            root=cfg.DIR_VECOLOR,
            phase='val',
            transform=val_transforms
        )

        if len(train_datasets) > 1:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        elif len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            self.print_log('No dataset!')
            exit(0)

        val_dataset = val_vecolor_dataset

        if cfg.DIST_ENABLE:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=1,
            rank=0)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=1,
            rank=0)

        self.trainloader = DataLoader(
            train_dataset,
            batch_size=int(cfg.TRAIN_BATCH_SIZE / cfg.TRAIN_GPUS),
            shuffle=False,
            num_workers=cfg.DATA_WORKERS, 
            pin_memory=True, 
            sampler=self.train_sampler)
        self.valloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0, 
            pin_memory=True, 
            sampler=self.val_sampler)

        self.print_log('Done!')

    def training(self):
        
        cfg = self.cfg

        running_losses = AverageMeter()
        running_accs = [AverageMeter() for _ in range(4)]

        batch_time = AverageMeter()
        avg_obj =  AverageMeter()       

        optimizer = self.optimizer
        criterion = self.criterion
        model = self.dist_model
        train_sampler = self.train_sampler
        trainloader = self.trainloader
        step = self.step
        epoch = self.epoch
        max_itr = cfg.TRAIN_TOTAL_STEPS

        self.print_log('Start training.')
        model.train()
        while step < cfg.TRAIN_TOTAL_STEPS:
            train_sampler.set_epoch(epoch)
            epoch += 1
            last_time = time.time()
            for sample_id, sample in enumerate(trainloader):
                now_lr = adjust_learning_rate(
                    optimizer=optimizer, 
                    base_lr=cfg.TRAIN_LR, 
                    p=cfg.TRAIN_POWER, 
                    itr=step, 
                    max_itr=max_itr, 
                    warm_up_steps=cfg.TRAIN_WARM_UP_STEPS, 
                    is_cosine_decay=cfg.TRAIN_COSINE_DECAY)

                images = sample['image'].cuda(self.gpu)
                labels = sample['label'].cuda(self.gpu)
                labels = labels.squeeze(1)

                preds = model(images)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN_CLIP_GRAD_NORM)
                optimizer.step()
                
                accs = pytorch_acc(preds, labels)
                
                running_losses.update(loss.item())
                for idx in range(len(accs)):
                    running_accs[idx].update(accs[idx])

                batch_time.update(time.time() - last_time)
                last_time = time.time()

                if step % cfg.TRAIN_LOG_STEP == 0 and self.rank == 0:
                    strs = 'Itr:{:06d}, LR:{:.6f}, Time:{:.3f}'.format(step, now_lr, batch_time.avg)
                    batch_time.reset()

                    strs += ', Loss {:.3f}({:.3f}) Acc/P/R/F1: {:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(
                                running_losses.val, running_losses.avg, 
                                running_accs[0].val, running_accs[1].val, running_accs[2].val,running_accs[3].val,)
                    running_losses.reset()
                    for idx in range(4):
                        running_accs[idx].reset()

                    self.print_log(strs)
                
                if step % cfg.TRAIN_EVAL_STEPS == 0 or step == 0:
                    self.print_log("Eval iter {:06d}...".format(step))
                    self.eval_training()

                if step % cfg.TRAIN_SAVE_STEP == 0 and step != 0 and self.rank == 0:
                    self.print_log('Save CKPT (Step {}).'.format(step))
                    save_network(self.model, optimizer, step, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT)

                step += 1
                if step > cfg.TRAIN_TOTAL_STEPS:
                    break

        if self.rank == 0:
            self.print_log('Save final CKPT (Step {}).'.format(step - 1))
            save_network(self.model, optimizer, step - 1, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT)

    def eval_training(self):
        cfg = self.cfg
        model = self.model
        criterion = self.criterion

        running_losses = AverageMeter()
        running_accs = [AverageMeter() for _ in range(4)]
        
        with torch.no_grad():
            model.eval()
            for sample_id, sample in enumerate(self.valloader):
                images = sample['image'].cuda(self.gpu)
                labels = sample['label'].cuda(self.gpu)
                labels = labels.squeeze(1)

                preds = model(images)
                loss = criterion(preds, labels)                
                accs = pytorch_acc(preds, labels)
                
                running_losses.update(loss.item())
                for idx in range(len(accs)):
                    running_accs[idx].update(accs[idx])
            
            strs = 'Val Loss: {:.3f} Acc/P/R/F1: {:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(
                            running_losses.avg, 
                            running_accs[0].avg, running_accs[1].avg, running_accs[2].avg, running_accs[3].avg,)
            self.print_log(strs)
        model.train()

    def print_log(self, string):
        gct = lambda: time.strftime('[%m/%d %H:%M:%S]', time.localtime(time.time()))
        if self.rank == 0:
            print("{} {}".format(gct(), string))



