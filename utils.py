import numpy as np
import torch
import yaml
import random
import torch.nn.functional as F

from monai.utils import set_determinism
from pathlib import Path
from monai.networks.nets import UNet
from monai.transforms import MapTransform


class ConvertToMultiChannelMaskd(MapTransform):
    """
        Convert multi-label singe-channel mask to multiple-channel one-hot encoded 
    """
    def __call__(self, data):
        d = dict(data)
        # assuming label values are 0, 1, 2, 4
        background = d['label'][0] == 0
        non_enhancing_tumor = d['label'][0] == 1
        edema = d['label'][0] == 2
        enhancing_tumor = d['label'][0] == 4
        d['label'] = torch.stack((background, non_enhancing_tumor, edema, enhancing_tumor), axis=0).astype(torch.float32)

        return d
    

class AddLastDimBIRADS34d(MapTransform):
    """
    """
    def __call__(self, data):
        d = dict(data)
        if d['image'].ndim == 3:
            if d['image'].shape[2] == 1:
                d['image'] = d['image'].squeeze(-1)  # remove last dim if it's 1
            elif d['image'].shape[0] == 1:
                d['image'] = d['image'].squeeze(0)  # remove first dim if it's 1
                
        # if 'birads_34' in d['image'].meta['filename_or_obj']:
        #     d['image'] = d['image'].unsqueeze(-1)  # add dim at the end
        # add channel dim if not present
        if d['image'].ndim == 2:
            d['image'] = d['image'].unsqueeze(0)  # add channel dim at the beginning

        return d


def evaluate_true_false(inp):
    inp = str(inp).upper()
    if 'TRUE'.startswith(inp):
        return True
    elif 'FALSE'.startswith(inp):
        return False
    else:
        raise ValueError('Argument error. Expected bool type.')


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_file_path):
    with open(config_file_path, "w") as f:
        yaml.dump(config, f)
    return
    

def set_seed(manual_seed=0):
    random.seed(manual_seed)
    np.random.seed(seed=manual_seed)
    set_determinism(seed=manual_seed)
    torch.manual_seed(seed=manual_seed)
    return


def pad_to(t, H, W):
    _, h, w = t.shape
    return F.pad(t, (0, W-w, 0, H-h), value=0)


def collate_pad(batch):
    imgs = [b['image'] for b in batch]
    # enforce tensor & channels
    imgs = [img.float() for img in imgs]
    max_h = max(i.shape[1] for i in imgs)
    max_w = max(i.shape[2] for i in imgs)
    imgs = [pad_to(i, max_h, max_w) for i in imgs]
    images = torch.stack(imgs, dim=0)
    out = {'image': images}
    # collate other fields with default behavior
    for k in batch[0].keys():
        if k != 'image':
            out[k] = torch.utils.data._utils.collate.default_collate([b[k] for b in batch])
    return out
