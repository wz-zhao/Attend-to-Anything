
import os
from os.path import join
import csv
import cv2, copy
import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import sys
from scipy.io import wavfile
import random
import json
import pandas as pd
import struct
import torch.nn.functional as F

def safe_listdir(dir_path, exts=None):

    names = []
    try:
        for n in os.listdir(dir_path):
            if n.startswith('.'):
                continue
            p = os.path.join(dir_path, n)
            if not os.path.isfile(p):
                continue
            if exts is not None and (not n.lower().endswith(exts)):
                continue
            names.append(n)
    except FileNotFoundError:
        return []
    names.sort()
    return names


class DHF1KDataset(Dataset):
    def __init__(
        self,
        path_data,
        len_snippet,
        epoch=0,
        mode="train",
        multi_frame=0,
        alternate=1,
                      
        starts_per_video_range=(3, 4), 
                                                                               
        alternate_choices=(1, 2, 3, 4),                       
        hflip_prob=0.5,                               
        temporal_reverse_prob=0.5,                    
        out_size=512,
        skip_first_n=5,
    ):
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.multi_frame = multi_frame
        self.alternate = alternate                      
        self.epoch = epoch
        self.dataset_name = "DHF1K"

        self.starts_per_video_range = starts_per_video_range
        self.alternate_choices = tuple(alternate_choices)
        self.hflip_prob = hflip_prob
        self.temporal_reverse_prob = temporal_reverse_prob
        self.out_size = out_size
        self.skip_first_n = int(skip_first_n)

        self.img_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        video_root = os.path.join(path_data, "video")
        self.video_names = [
            v for v in sorted(os.listdir(video_root))
            if (not v.startswith('.')) and os.path.isdir(os.path.join(video_root, v))
        ]

        self.video_frame_counts = {}
        for v in self.video_names:
            img_dir = os.path.join(video_root, v, "images")
                                              
            self.video_frame_counts[v] = len(safe_listdir(img_dir, exts=('.png', '.jpg', '.jpeg', '.bmp')))


                                        
        if self.mode == "train":
            self._build_train_index()
        else:
            self._build_eval_index()

                                                                            
    def _last_start(self, num_frames: int, alternate: int) -> int:
                                                                  
                                
        # => start_idx <= num_frames - (alternate*(len_snippet-1) + 1)
        return num_frames - (alternate * (self.len_snippet - 1) + 1)

    def _can_sample(self, num_frames: int, alternate: int) -> bool:
        return self._last_start(num_frames, alternate) >= self.skip_first_n

                                                               
    def _build_train_index(self):
        self.index = []
        lo, hi = self.starts_per_video_range

        for v in self.video_names:
            n = self.video_frame_counts[v]

                                    
            if not any(self._can_sample(n, a) for a in self.alternate_choices):
                continue

            k = random.randint(lo, hi)
            for _ in range(k):
                self.index.append(v)

        random.shuffle(self.index)

                                                          
    def _build_eval_index(self):
        self.index = []
        for v in self.video_names:
            n = self.video_frame_counts[v]
            if not self._can_sample(n, self.alternate):
                continue

            last_start = self._last_start(n, self.alternate)
            start0 = self.skip_first_n

                                              
            for i in range(start0, last_start + 1, self.len_snippet):
                self.index.append((v, i))

                                       
            if not self.index or self.index[-1] != (v, last_start):
                self.index.append((v, last_start))

                                    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.mode == "train":
            random.seed(epoch)
            np.random.seed(epoch)
            self._build_train_index()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.mode == "train":
            file_name = self.index[idx]
            num_frames = self.video_frame_counts[file_name]

                                             
            valid_alts = [a for a in self.alternate_choices if self._can_sample(num_frames, a)]
            if len(valid_alts) == 0:
                                                     
                valid_alts = [1] if self._can_sample(num_frames, 1) else [min(self.alternate_choices)]

            alternate = random.choice(valid_alts)

            min_start = self.skip_first_n
            max_start = self._last_start(num_frames, alternate)
            start_idx = np.random.randint(min_start, max_start + 1)

        else:
            file_name, start_idx = self.index[idx]
            alternate = self.alternate

                                                    
            start_idx = max(start_idx, self.skip_first_n)

        path_clip = os.path.join(self.path_data, "video", file_name, "images")
        path_annt = os.path.join(self.path_data, "annotation", file_name, "maps")
        path_fix  = os.path.join(self.path_data, "annotation", file_name, "fixation")

        num_frames = self.video_frame_counts[file_name]
    
        clip_img, clip_gt, clip_fix = [], [], []

        for i in range(self.len_snippet):
                                      
            frame_idx = start_idx + alternate * i + 1
            if frame_idx > num_frames:
                frame_idx = num_frames                          

            img = Image.open(os.path.join(path_clip, f"{frame_idx:04d}.png")).convert("RGB")
            clip_img.append(self.img_transform(img))

            if self.mode != "save":
                # ====== maps ======
                gt = np.array(Image.open(os.path.join(path_annt, f"{frame_idx:04d}.png")).convert("L"))
                if self.mode == "train":
                    gt = cv2.resize(gt.astype("float32"), (448, 448))
                else:
                    gt = gt.astype("float32")
                if gt.max() > 1.0:
                    gt = gt / 255.0
                clip_gt.append(torch.from_numpy(gt))

                # ====== fixation ======
                fx = np.array(Image.open(os.path.join(path_fix, f"{frame_idx:04d}.png")).convert("L"))
                
                fx = fx.astype("float32")

                if fx.max() > 1.0:
                    fx = fx / 255.0
                fx = (fx > 0.5).astype("float32")
                clip_fix.append(torch.from_numpy(fx))

        clip_fix = torch.stack(clip_fix, dim=0).unsqueeze(1)


        if self.mode == "save":
                               
            return clip_img, start_idx, file_name, img.size

        clip_gt = torch.stack(clip_gt, dim=0)    # [T, H, W]
        clip_img = torch.stack(clip_img, dim=0)  # [T, 3, H, W]
        
        

                            
        if self.mode == "train" and random.random() < self.hflip_prob:
            clip_img = torch.flip(clip_img, dims=[3]).contiguous()        
            clip_gt  = torch.flip(clip_gt,  dims=[2]).contiguous()        

                
        if self.mode == "train" and random.random() < self.temporal_reverse_prob:
            clip_img = torch.flip(clip_img, dims=[0]).contiguous()
            clip_gt  = torch.flip(clip_gt,  dims=[0]).contiguous()

        clip_gt = clip_gt.unsqueeze(1).float()  # [T, 1, H, W]


        return {
            "image": clip_img,
            "saliency": clip_gt,
            "fixation": clip_fix,
            "label": self.dataset_name
        }


    

import os, random, copy
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Hollywood_UCFDataset(Dataset):
    def __init__(
        self,
        path_data,
        len_snippet,
        name,
        epoch=0,
        mode="train",
        multi_frame=0,
        alternate=1,
               
        samples_per_video=None,
                          
        starts_per_video_range=(1, 1),
        # starts_per_video_range=(2, 3),
        alternate_choices=(1, 2, 3, 4),
        hflip_prob=0.5,
        temporal_reverse_prob=0.5,
        out_size=448,
    ):
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.multi_frame = multi_frame
        self.alternate = alternate
        self.epoch = epoch
        self.dataset_name = "Hollywood_UCF"

        self.starts_per_video_range = starts_per_video_range
        self.alternate_choices = alternate_choices
        self.hflip_prob = hflip_prob
        self.temporal_reverse_prob = temporal_reverse_prob
        self.out_size = out_size

        self.img_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

              
        self.video_names = sorted(os.listdir(path_data))

                                
        self.video_frames = {}
        self.video_maps = {}
        self.video_frame_counts = {}

        if samples_per_video is not None:
            starts_per_video_range = (int(samples_per_video), int(samples_per_video))
        self.starts_per_video_range = starts_per_video_range

        for v in self.video_names:
            img_dir = os.path.join(path_data, v, "images")
            gt_dir  = os.path.join(path_data, v, "maps")
            if (not os.path.isdir(img_dir)) or (not os.path.isdir(gt_dir)):
                continue

            frames = safe_listdir(img_dir, exts=('.png', '.jpg', '.jpeg', '.bmp'))
            maps   = safe_listdir(gt_dir,  exts=('.png', '.jpg', '.jpeg', '.bmp'))


            if len(frames) == 0 or len(maps) == 0:
                continue

                                                 
            self.video_frames[v] = frames
            self.video_maps[v] = maps
            self.video_frame_counts[v] = len(frames)

        self.video_names = sorted(list(self.video_frames.keys()))
        
        
                               
#         if self.mode in ["train"]:
#             self.video_names = self.video_names[:200]

                                       
#         self.video_frames = {k: self.video_frames[k] for k in self.video_names}
#         self.video_maps = {k: self.video_maps[k] for k in self.video_names}
#         self.video_frame_counts = {k: self.video_frame_counts[k] for k in self.video_names}

    

             
        if self.mode == "train":
            self._build_train_index()
        else:
            self.index = []
            for v in self.video_names:
                n = self.video_frame_counts[v]
                if n < self.alternate * self.len_snippet:
                    continue
                if self.mode == "val":
                    for i in range(0, n - self.alternate * self.len_snippet, self.len_snippet):
                        self.index.append((v, i))
                    self.index.append((v, n - self.alternate * self.len_snippet))
                elif self.mode == "test":
                    for i in range(0, n - self.alternate * self.len_snippet + 1):
                        self.index.append((v, i))
                    self.index.append((v, n - self.alternate * self.len_snippet))

    def _build_train_index(self):
        self.index = []
        lo, hi = self.starts_per_video_range
        for v in self.video_names:
            n = self.video_frame_counts[v]
            if n < 1 * self.len_snippet:
                continue
            k = random.randint(lo, hi)
            for _ in range(k):
                self.index.append(v)
        random.shuffle(self.index)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.mode == "train":
            random.seed(epoch)
            np.random.seed(epoch)
            self._build_train_index()

    def __len__(self):
        return len(self.index)

    def _pad_to_len(self, frames, maps):
        if len(frames) < self.len_snippet:
            frames = [frames[0]] * (self.len_snippet - len(frames)) + frames
        if len(maps) < self.len_snippet:
            maps = [maps[0]] * (self.len_snippet - len(maps)) + maps
        return frames, maps

    def __getitem__(self, idx):
        if self.mode == "train":
            file_name = self.index[idx]
            alternate = random.choice(self.alternate_choices)

            num_frames = self.video_frame_counts[file_name]
            max_start = max(1, num_frames - alternate * self.len_snippet + 1)
            start_idx = np.random.randint(0, max_start)
        else:
            file_name, start_idx = self.index[idx]
            alternate = self.alternate

        path_clip = os.path.join(self.path_data, file_name, "images")
        path_annt = os.path.join(self.path_data, file_name, "maps")
        path_fix  = os.path.join(self.path_data, file_name, "fixation")
        

        frames = self.video_frames[file_name]
        maps   = self.video_maps[file_name]
        frames, maps = self._pad_to_len(frames, maps)

        clip_img, clip_gt, clip_fix = [], [], []

        last_frame_idx = len(frames) - 1
        last_map_idx   = len(maps) - 1

        
        for i in range(self.len_snippet):
            fidx = min(start_idx + alternate * i, last_frame_idx)
            midx = min(start_idx + alternate * i, last_map_idx)

            img = Image.open(os.path.join(path_clip, frames[fidx])).convert("RGB")
            clip_img.append(self.img_transform(img))

            gt = np.array(Image.open(os.path.join(path_annt, maps[midx])).convert("L"))
            if self.mode == "train":
                gt = cv2.resize(gt.astype("float32"), (448, 448))
            else:
                gt = gt.astype("float32")
        
            if gt.max() > 1.0:
                gt = gt / 255.0
            clip_gt.append(torch.from_numpy(gt))
            
            fx = np.array(Image.open(os.path.join(path_fix, frames[fidx])).convert("L"))
            fx = fx.astype("float32")

            if fx.max() > 1.0:
                fx = fx / 255.0
            fx = (fx > 0.5).astype("float32")
            clip_fix.append(torch.from_numpy(fx))

        clip_fix = torch.stack(clip_fix, dim=0).unsqueeze(1)

        
        clip_img = torch.stack(clip_img, dim=0)  # [T, 3, H, W]
        clip_gt  = torch.stack(clip_gt,  dim=0)  # [T, H, W]


        if self.mode == "train" and random.random() < self.hflip_prob:
            clip_img = torch.flip(clip_img, dims=[3]).contiguous()
            clip_gt  = torch.flip(clip_gt,  dims=[2]).contiguous()

        if self.mode == "train" and random.random() < self.temporal_reverse_prob:
            clip_img = torch.flip(clip_img, dims=[0]).contiguous()
            clip_gt  = torch.flip(clip_gt,  dims=[0]).contiguous()

        clip_gt = clip_gt.unsqueeze(1).float()  # [T, 1, H, W]

        return {
            "image": clip_img,
            "saliency": clip_gt,
            "fixation": clip_fix,
            "label": self.dataset_name}

    
import os, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def safe_listdir(d, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(exts):
            out.append(f)
    return out


def load_ledov_datamat(mat_path: str):
                  
    try:
        import h5py

        with h5py.File(mat_path, "r") as f:
            g = f["Data"]
            fps = float(np.array(g["VideoFrameRate"])[0, 0])
            num_frames = int(np.array(g["VideoFrames"])[0, 0])

            size = np.array(g["VideoSize"]).reshape(-1)
            vid_w, vid_h = int(size[0]), int(size[1])                                          

            fix = np.array(g["fixdata"])
            if fix.shape[0] == 5:
                fix = fix.T
            fix = fix.astype(np.float32)

        return fps, num_frames, (vid_w, vid_h), fix

    except Exception:
        # fallback: scipy (Matlab <= v7)
        from scipy.io import loadmat

        m = loadmat(mat_path)
        Data = m["Data"]

                                                
        fps = float(Data["VideoFrameRate"][0, 0][0, 0])
        num_frames = int(Data["VideoFrames"][0, 0][0, 0])

        size = Data["VideoSize"][0, 0].reshape(-1)
        vid_w, vid_h = int(size[0]), int(size[1])

        fix = Data["fixdata"][0, 0]
        if fix.shape[0] == 5:
            fix = fix.T
        fix = fix.astype(np.float32)

        return fps, num_frames, (vid_w, vid_h), fix


def build_fixation_for_clip_matlogic(
    fixdata: np.ndarray,
    fps: float,
    vid_w: int,
    vid_h: int,
    clip_frame_indices_0based: list,
    out_size: int,
):
   
    T = len(clip_frame_indices_0based)
    per_frame_xy = [np.zeros((0, 2), np.float32) for _ in range(T)]
    fixmap = np.zeros((T, out_size, out_size), np.float32)

    if fixdata is None or fps is None or vid_w is None or vid_h is None:
        return per_frame_xy, torch.from_numpy(fixmap).unsqueeze(1)

    frame_duration_ms = 1000.0 / float(fps)
    centermask = round(vid_h / 20.0)

    frame_to_t = {f: t for t, f in enumerate(clip_frame_indices_0based)}

    beginflag = True
    prev_start = None

    for k in range(fixdata.shape[0]):
        start_ms = float(fixdata[k, 1])  # MATLAB col2
        dur_ms   = float(fixdata[k, 2])  # MATLAB col3
        vx       = float(fixdata[k, 3])  # MATLAB col4
        vy       = float(fixdata[k, 4])  # MATLAB col5

        if prev_start is None or start_ms < prev_start:
            beginflag = True
        prev_start = start_ms

                           
        if not (0 < vx < vid_w and 0 < vy < vid_h):
            continue

                                                                 
        if beginflag:
            outside_center = (abs(vx - vid_w / 2.0) > centermask) or (abs(vy - vid_h / 2.0) > centermask)
            if not outside_center:
                continue
            beginflag = False

        start_frame_1 = int(np.ceil(start_ms / frame_duration_ms))
        end_frame_1   = int(np.ceil((start_ms + dur_ms) / frame_duration_ms))
        if start_frame_1 <= 0:
            start_frame_1 = 1

        start_f = start_frame_1 - 1  # -> 0-based
        end_f   = end_frame_1 - 1

                        
        sx = vx * (out_size / float(vid_w))
        sy = vy * (out_size / float(vid_h))

        for f in range(start_f, end_f + 1):
            if f in frame_to_t:
                t = frame_to_t[f]
                per_frame_xy[t] = np.concatenate(
                    [per_frame_xy[t], np.array([[sx, sy]], np.float32)],
                    axis=0
                )
                ix, iy = int(round(sx)), int(round(sy))
                if 0 <= ix < out_size and 0 <= iy < out_size:
                    fixmap[t, iy, ix] = 1.0

    return per_frame_xy, torch.from_numpy(fixmap).unsqueeze(1)


class LEDOVDataset(Dataset):
    def __init__(
        self,
        path_data,
        len_snippet,
        video_name_list=None,
        epoch=0,
        mode="train",
        multi_frame=0,
        alternate=1,
        starts_per_video_range=(1, 2),
        alternate_choices=(1, 2, 3, 4),
        hflip_prob=0.5,
        temporal_reverse_prob=0.5,
        out_size=448,
        image_path="datasets/LEDOV/image",
        saliency_path="datasets/LEDOV/mask",
        fixation_path="datasets/LEDOV/fixation",
    ):
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.multi_frame = multi_frame
        self.alternate = alternate
        self.epoch = epoch
        self.dataset_name = "LEDOV"

        self.video_name_list = video_name_list if video_name_list is not None else []
        self.image_path = image_path
        self.saliency_path = saliency_path
        self.fixation_path = fixation_path

        self.starts_per_video_range = starts_per_video_range
        self.alternate_choices = alternate_choices
        self.hflip_prob = hflip_prob
        self.temporal_reverse_prob = temporal_reverse_prob
        self.out_size = out_size

        self.img_transform = transforms.Compose([
            transforms.Resize((out_size, out_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

              
        if self.video_name_list:
            self.video_names = list(self.video_name_list)
        else:
            self.video_names = sorted(os.listdir(self.image_path))

                    
        self.video_frames = {}
        self.video_maps = {}
        self.video_frame_counts = {}

        for v in self.video_names:
            img_dir = os.path.join(self.image_path, v)
            gt_dir  = os.path.join(self.saliency_path, v)

            if (not os.path.isdir(img_dir)) or (not os.path.isdir(gt_dir)):
                continue

            frames = safe_listdir(img_dir)
            maps   = safe_listdir(gt_dir)
            if len(frames) == 0 or len(maps) == 0:
                continue

            self.video_frames[v] = frames
            self.video_maps[v] = maps
            self.video_frame_counts[v] = len(frames)

        self.video_names = sorted(list(self.video_frames.keys()))

                                                            
        self.video_fixdata = {}
        self.video_fps = {}
        self.video_size = {}

        for v in self.video_names:
            mat_path = os.path.join(self.fixation_path, v, "Data.mat")
            if os.path.isfile(mat_path):
                fps, nframes, (vid_w, vid_h), fixdata = load_ledov_datamat(mat_path)
                self.video_fps[v] = fps
                self.video_size[v] = (vid_w, vid_h)
                self.video_fixdata[v] = fixdata
            else:
                self.video_fps[v] = None
                self.video_size[v] = None
                self.video_fixdata[v] = None

             
        if self.mode == "train":
            self._build_train_index()
        else:
            self.index = []
            for v in self.video_names:
                n = self.video_frame_counts[v]
                if n < self.alternate * self.len_snippet:
                    continue
                if self.mode == "val":
                    for i in range(0, n - self.alternate * self.len_snippet, self.len_snippet):
                        self.index.append((v, i))
                    self.index.append((v, n - self.alternate * self.len_snippet))
                elif self.mode == "test":
                    for i in range(0, n - self.alternate * self.len_snippet + 1):
                        self.index.append((v, i))
                    self.index.append((v, n - self.alternate * self.len_snippet))

    def _build_train_index(self):
        self.index = []
        lo, hi = self.starts_per_video_range
        for v in self.video_names:
            n = self.video_frame_counts[v]
            if n < 1 * self.len_snippet:
                continue
            k = random.randint(lo, hi)
            for _ in range(k):
                self.index.append(v)
        random.shuffle(self.index)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.mode == "train":
            random.seed(epoch)
            np.random.seed(epoch)
            self._build_train_index()

    def __len__(self):
        return len(self.index)

    def _pad_to_len(self, frames, maps):
        if len(frames) < self.len_snippet:
            frames = [frames[0]] * (self.len_snippet - len(frames)) + frames
        if len(maps) < self.len_snippet:
            maps = [maps[0]] * (self.len_snippet - len(maps)) + maps
        return frames, maps

    def __getitem__(self, idx):
        if self.mode == "train":
            file_name = self.index[idx]
            alternate = random.choice(self.alternate_choices)

            num_frames = self.video_frame_counts[file_name]
            max_start = max(1, num_frames - alternate * self.len_snippet + 1)
            start_idx = np.random.randint(0, max_start)
        else:
            file_name, start_idx = self.index[idx]
            alternate = self.alternate

        path_clip = os.path.join(self.image_path, file_name)
        path_annt = os.path.join(self.saliency_path, file_name)

        frames = self.video_frames[file_name]
        maps   = self.video_maps[file_name]
        frames, maps = self._pad_to_len(frames, maps)

        last_frame_idx = len(frames) - 1
        last_map_idx   = len(maps) - 1

                                 
        clip_frame_indices = []
        for i in range(self.len_snippet):
            fidx = min(start_idx + alternate * i, last_frame_idx)
            clip_frame_indices.append(fidx)

        # ===== fixation from Data.mat (MAT logic) =====
        fixdata = self.video_fixdata.get(file_name, None)
        fps = self.video_fps.get(file_name, None)
        size = self.video_size.get(file_name, None)
        if size is None:
            vid_w, vid_h = self.out_size, self.out_size
        else:
            vid_w, vid_h = size

        per_frame_xy, clip_fixmap = build_fixation_for_clip_matlogic(
            fixdata=fixdata,
            fps=fps,
            vid_w=vid_w,
            vid_h=vid_h,
            clip_frame_indices_0based=clip_frame_indices,
            out_size=self.out_size,
        )

        # ===== read image & saliency =====
        clip_img, clip_gt = [], []
        for i in range(self.len_snippet):
            fidx = clip_frame_indices[i]
            midx = min(start_idx + alternate * i, last_map_idx)

            img = Image.open(os.path.join(path_clip, frames[fidx])).convert("RGB")
            clip_img.append(self.img_transform(img))

            gt = np.array(Image.open(os.path.join(path_annt, maps[midx])).convert("L"))
            gt = gt.astype("float32")
            if self.mode == "train":
                gt = cv2.resize(gt, (self.out_size, self.out_size))

            if gt.max() > 1.0:
                gt = gt / 255.0
            clip_gt.append(torch.from_numpy(gt))

        clip_img = torch.stack(clip_img, dim=0)                 # [T,3,H,W]
        clip_gt  = torch.stack(clip_gt,  dim=0).unsqueeze(1)    # [T,1,H,W]

        # ===== augmentation (sync fixation) =====
        if self.mode == "train" and random.random() < self.hflip_prob:
            clip_img = torch.flip(clip_img, dims=[3]).contiguous()
            clip_gt  = torch.flip(clip_gt,  dims=[3]).contiguous()
            clip_fixmap = torch.flip(clip_fixmap, dims=[3]).contiguous()

            W = self.out_size
            per_frame_xy = [
                (np.stack([W - 1 - xy[:, 0], xy[:, 1]], axis=1).astype(np.float32) if xy.shape[0] > 0 else xy)
                for xy in per_frame_xy
            ]

        if self.mode == "train" and random.random() < self.temporal_reverse_prob:
            clip_img = torch.flip(clip_img, dims=[0]).contiguous()
            clip_gt  = torch.flip(clip_gt,  dims=[0]).contiguous()
            clip_fixmap = torch.flip(clip_fixmap, dims=[0]).contiguous()
            per_frame_xy = list(reversed(per_frame_xy))
            clip_frame_indices = list(reversed(clip_frame_indices))

        fixation = {
            "per_frame_xy": per_frame_xy,              # list len=T, each (Ni,2) in resized coords
            "fixmap": clip_fixmap,                     # Tensor [T,1,H,W]
            "clip_frame_indices": clip_frame_indices,  # list len=T, 0-based original frame ids
            "fps": fps,
            "video_size": (vid_w, vid_h),
        }

        return {
            "image": clip_img,
            "saliency": clip_gt,
            "fixation": fixation,
            "label": self.dataset_name,
        }




import os
import shutil
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def safe_listdir(path, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    if not os.path.isdir(path):
        return []
    out = []
    for n in os.listdir(path):
        if n.startswith("."):
            continue
        p = os.path.join(path, n)
        if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
            out.append(n)
    out.sort()
    return out


class DHF1KDataset_save(Dataset):

    def __init__(
        self,
        path_data,                     
        len_snippet=32,                    
        mode="save",
        out_hw=(448, 448),
        old_img_root="datasets/DHF1K/DHF1K_val/video",
        new_img_root="datasets/DHF1K/DHF1K_save",

        copy_to_new_root=True,
        copy_backend="copy2",           
        exts=(".png", ".jpg", ".jpeg", ".bmp"),
    ):
        assert mode != "train", "这个类是 save/test 用的（无GT逐帧遍历），不用于 train。"
        self.path_data = path_data.rstrip("/")
        self.len_snippet = int(len_snippet)
        self.mode = mode
        self.dataset_name = "DHF1K"

        self.old_img_root = old_img_root.rstrip("/")
        self.new_img_root = new_img_root.rstrip("/")
        self.copy_to_new_root = bool(copy_to_new_root)
        self.copy_backend = copy_backend
        self.exts = exts

        self.img_transform = transforms.Compose([
            transforms.Resize(out_hw),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

                                                                  
        self.video_names = [
            v for v in sorted(os.listdir(self.path_data))
            if (not v.startswith(".")) and os.path.isdir(os.path.join(self.path_data, v))
        ]

                            
        self.video_frame_files = {}  # v -> ["0001.png", ...]
        self.index = []                                                                                

        for v in self.video_names:
            img_dir = os.path.join(self.path_data, v, "images")
            files = safe_listdir(img_dir, exts=self.exts)
            if len(files) == 0:
                continue
            self.video_frame_files[v] = files

            n = len(files)

            if n < self.len_snippet:
                                                 
                self.index.append((v, 0))
            else:
                last_start = n - self.len_snippet                 

                                        
                stride = self.len_snippet
                for s in range(0, last_start + 1, stride):
                    self.index.append((v, s))

                                                              
                                           
                if self.index[-1] != (v, last_start):
                    self.index.append((v, last_start))


    def __len__(self):
        return len(self.index)

    def _map_to_save_root(self, src_path: str) -> str:
        src = src_path.replace("\\", "/")
        old = self.old_img_root.replace("\\", "/")
        new = self.new_img_root.replace("\\", "/")
        if src.startswith(old):
            return src.replace(old, new, 1)
                       
        return os.path.join(new, src.lstrip("/"))

    def _copy_if_needed(self, src: str, dst: str):
        if not self.copy_to_new_root:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            return
        try:
            if self.copy_backend == "copy2":
                shutil.copy2(src, dst)
            elif self.copy_backend == "copy":
                shutil.copy(src, dst)
            elif self.copy_backend == "link":
                os.link(src, dst)
            elif self.copy_backend == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
        except FileExistsError:
            pass
        except OSError:
                                       
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

    def __getitem__(self, idx):
        v, start = self.index[idx]
        files = self.video_frame_files[v]            # list of filenames
        img_dir = os.path.join(self.path_data, v, "images")
        n = len(files)

                                      
        if n >= self.len_snippet:
            clip_files = files[start:start + self.len_snippet]
        else:
            clip_files = files[:] + [files[-1]] * (self.len_snippet - n)

        clip_img = []
        saved_paths = []
        src_paths = []

        for fn in clip_files:
            src_img_path = os.path.join(img_dir, fn)
            dst_img_path = self._map_to_save_root(src_img_path)

            self._copy_if_needed(src_img_path, dst_img_path)

            img = Image.open(src_img_path).convert("RGB")
            clip_img.append(self.img_transform(img))

            src_paths.append(src_img_path)
            saved_paths.append(dst_img_path)

        clip_img = torch.stack(clip_img, dim=0)  # [T,3,448,448]

        return {
            "image": clip_img,
            "video": v,
            "start_pos": int(start),     # 0-based start index in this video
            "frame_paths": saved_paths,                                 
            "src_frame_paths": src_paths,
            "label": self.dataset_name,
        }

    
    
  
