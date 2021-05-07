import sys
sys.path.append('.')
sys.path.append('..')
from networks.engine.train_manager import Trainer
import torch.multiprocessing as mp
import importlib

def main_worker(gpu, cfg):
    # Initiate a training manager
    trainer = Trainer(rank=gpu, cfg=cfg)
    # Start Training
    trainer.training()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Vehicle Color Recognition.")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--config', type=str, default='configs.res2net_baseline')

    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)

    parser.add_argument('--pretrained_path', type=str, default='')

    parser.add_argument('--datasets', nargs='+', type=str, default=['youtubevos'])
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1.)
    parser.add_argument('--start_step', type=int, default=-1.)

    args = parser.parse_args()

    config = importlib.import_module(args.config)
    cfg = config.cfg
    
    if args.exp_name != '':
        cfg.EXP_NAME = args.exp_name

    cfg.DIST_START_GPU = args.start_gpu
    if args.gpu_num > 0:
        cfg.TRAIN_GPUS = args.gpu_num
    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.pretrained_path != '':
        cfg.PRETRAIN_MODEL = args.pretrained_path

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr
    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step
        cfg.TRAIN_START_SEQ_TRAINING_STEPS = int(args.total_step / 2)
        cfg.TRAIN_HARD_MINING_STEP = int(args.total_step / 2)
    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step

    # Use torch.multiprocessing.spawn to launch distributed processes
    if cfg.DIST_ENABLE:
        mp.spawn(main_worker, nprocs=cfg.TRAIN_GPUS, args=(cfg,))
    else:
        main_worker(gpu=0, cfg=cfg)

if __name__ == '__main__':
    main()

