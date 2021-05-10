import os, sys
import random
import numpy as np
import json
import cv2
import pprint

import torch
from torch.utils import data
CURR_DIR = os.path.dirname(__file__)

# if __name__ == "__main__": sys.path.insert(0, os.path.join(CURR_DIR, ".."))
# from dataset import transforms as tr

"""
VERI 数据集 车辆颜色&类型数据集的DataLoader
"""

random.seed(0); np.random.seed(0); torch.manual_seed(0)

from zqlib import readvdnames

class VECOLOR(data.Dataset):
    def __init__(self, root, phase, transform=None):
        self.root = root
        self.img_d = os.path.join(root, "JPEGImages")
        self.set_d = os.path.join(root, "Annotations")
        self.phase = phase
        self.color_list = COLOR_LIST
        assert self.phase in ["train", "val"], f"Phase {phase} must be `train`/`val`!"

        _set_f = os.path.join(self.set_d, "{}.json".format(phase))
        with open(_set_f, "r") as f:
            dset_meta = json.load(f)
        
        flatten_meta = []
        for cid in self.color_list:
            for sid in dset_meta[cid]:
                flatten_meta.append((cid, sid))
                
        self.dset_meta = dset_meta
        self.flatten_meta = flatten_meta
        self.dset_len = len(flatten_meta)
        
        self.transform = transform
    
        pprint.pprint({
            'root': root,
            'phase': self.phase,
            'distribution': [(k,len(v)) for k,v in dset_meta.items()],
        }, indent=1)
    
    def __len__(self):
        return len(self.flatten_meta)

    def __getitem__(self, index):
        if self.phase == "train":
            ret = self.get_train_item(index)
        else:
            ret = self.get_val_item(index)
        return ret

    def get_val_item(self, index):
        cid, img_id = self.flatten_meta[index]
        label = self.color_list.index(cid)

        img_path = os.path.join(self.img_d, cid, img_id)
        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
            label = torch.Tensor([label]).long()

        ret = {
            "image": img,
            "label": label,
            "path": img_path
        }
        return ret

    def get_train_item(self, index):
        rand_cid = random.choice(self.color_list)
        
        img_id = random.choice(self.dset_meta[rand_cid])
        label = self.color_list.index(rand_cid)
        
        img_path = os.path.join(self.img_d, rand_cid, img_id)
        img = cv2.imread(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
            label = torch.Tensor([label]).long()

        ret = {
            "image": img,
            "label": label,
            "path": img_path
        }
        return ret



if __name__ == "__main__":
    from torchvision import transforms 
    composed_transforms = transforms.Compose([
            tr.RandomScale(0.8, 1.3, 640),
            tr.RandomCrop(640),
            tr.RandomHorizontalFlip(0.5),
            tr.Resize(512),
            tr.ToTensor()])
    val_transforms = transforms.Compose([
            tr.Resize(512),
            tr.ToTensor()])

    ds_train = VECOLOR("../data", "train", transform=composed_transforms)
    ds_val = VECOLOR("../data", "val", transform=val_transforms)

    train_loader = data.DataLoader(dataset=ds_val, batch_size=1)

    def th2np(th, dtype='image', transpose=False, rgb_cyclic=False):
        assert dtype in ['image', 'mask']
        if dtype == 'image':
            th = (th + 1.0) / 2.0
            th = th * 255
            npdata = th.detach().cpu().numpy()      # NCHW
            if rgb_cyclic:
                npdata = npdata[:, ::-1, :, :]
            if transpose:
                npdata = npdata.transpose((0, 2, 3, 1)) # NHWC
        else:
            if th.ndim == 3:
                th = th.unsqueeze(1)
            if th.size(1) == 1:
                npdata = th.detach().cpu().repeat(1, 3, 1, 1).numpy()   # NCHW
            else:
                npdata = th.detach().cpu().numpy()
            if transpose:
                npdata = npdata.transpose((0, 2, 3, 1))
        return npdata

    label2name = tr.IdToLabel()

    for i, s in enumerate(train_loader):
        image = s["image"]
        label = s["label"]
        
        print(image.shape, label.shape)
        image_np = th2np(image, dtype='image', transpose=True, rgb_cyclic=False)
        image_np = np.transpose(image_np, (1, 0, 2, 3))
        image_np = image_np.reshape(512, -1, 3).copy()
        for ii in range(len(label)):
            cv2.putText(image_np, label2name(label[ii].item()), (20+ii*512, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), thickness=2)
        cv2.imwrite(f"image_{i}.png", image_np)

        print(image_np.shape)
        if i > 5:
            break


    exit()






















