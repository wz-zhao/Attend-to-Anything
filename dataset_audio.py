import os
from pathlib import Path
from os.path import join
import csv
import cv2, copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchaudio
import torchaudio.transforms as T            
import sys
from scipy.io import wavfile
import scipy.io as sio

import json
import random



def get_fixation_mat(path_data, dataset_name, video_name, frame_idx, key='eyeMap'):
  
    mat_path = join(path_data, 'annotations', dataset_name, video_name, f'fixMap_{frame_idx:05d}.mat')
    if not os.path.exists(mat_path):
        return None

    info = sio.loadmat(mat_path)
    if key not in info:
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {list(info.keys())}")

    fix = info[key]

                                                           
    fix = np.array(fix).squeeze().astype(np.float32)
    return fix



def get_frame_aligned_audio(audioind, audiodata, clip_length, start_idx, video_fps=30.0, target_sr=48000, context_sec=2.0):
 
    target_samples = int(target_sr * context_sec) # 48000 * 2 = 96000

                      
    if audioind not in audiodata:
        return torch.zeros(clip_length, 1, target_samples)

               
    raw_wav = audiodata[audioind]['wav'] # Tensor: (1, Total_Samples)
    orig_sr = audiodata[audioind]['Fs']
    
              
                                                                            
    frame_indices = np.arange(start_idx, start_idx + clip_length)
    
                                           
    # Time = Frame / FPS
    # Sample = Time * SR
    center_samples = (frame_indices / video_fps * orig_sr).astype(int)
    
    half_window = int((context_sec * orig_sr) / 2)             
    total_audio_len = raw_wav.shape[1]
    
    audio_segments = []

    for center in center_samples:
        start = center - half_window
        end = center + half_window
        
        pad_left = 0
        pad_right = 0
        
                        
        if start < 0:
            pad_left = -start
            start = 0
        if end > total_audio_len:
            pad_right = end - total_audio_len
            end = total_audio_len
            
            
        segment = raw_wav[:, start:end] # (1, current_len)
        
            
        if pad_left > 0 or pad_right > 0:
            segment = torch.nn.functional.pad(segment, (pad_left, pad_right))
            
                                   
        expected_len = int(orig_sr * context_sec)                        
        if segment.shape[1] != expected_len:
             segment = torch.nn.functional.interpolate(segment.unsqueeze(0), size=expected_len, mode='linear', align_corners=False).squeeze(0)

        audio_segments.append(segment)
        
    # Stack -> (T, 1, L_orig)
    audio_tensor = torch.stack(audio_segments, dim=0)

                                                
    if orig_sr != target_sr:
                
        resampler = T.Resample(orig_sr, target_sr)
        
                                              
        T_dim, C_dim, L_dim = audio_tensor.shape
        flat_audio = audio_tensor.view(T_dim, L_dim)            
        
               
        resampled = resampler(flat_audio) # (T, L_new)
        
                               
        audio_tensor = resampled.view(T_dim, C_dim, -1)

                                         
    if audio_tensor.shape[-1] != target_samples:
         audio_tensor = torch.nn.functional.interpolate(
             audio_tensor, size=target_samples, mode='linear', align_corners=False
         )

    return audio_tensor

def read_sal_text(txt_file):
    test_list = {'names': [], 'nframes': [], 'fps': []}
    with open(txt_file,'r') as f:
        for line in f:
            word=line.strip().split()
            test_list['names'].append(word[0])
            test_list['nframes'].append(word[1])
            test_list['fps'].append(word[2])
    return test_list

def read_sal_text_dave(json_file):
    test_list = {'names': [], 'nframes': [], 'fps': []}
    with open(json_file,'r') as f:
        _dic = json.load(f)
        for name in _dic:
            # word=line.strip().split()
            test_list['names'].append(name)
            test_list['nframes'].append(0)
            test_list['fps'].append(float(_dic[name]))
    return test_list    

def make_dataset(annotation_path, audio_path, gt_path, json_file=None):
    if json_file is None:
        data = read_sal_text(annotation_path)
    else:
        data = read_sal_text_dave(json_file)
    video_names = data['names']
    video_nframes = data['nframes']
    video_fps = data['fps']
    dataset = []
    audiodata= {}
    for i in range(len(video_names)):
        # if i % 100 == 0:
        #   # print('dataset loading [{}/{}]'.format(i, len(video_names)))

        n_frames = len(os.listdir(join(gt_path, video_names[i], 'maps')))
        # if n_frames <= 1:
        #   # print("Less frames")
        #   # continue

        begin_t = 1
        end_t = n_frames

        audio_wav_path = os.path.join(audio_path,video_names[i],video_names[i]+'.wav')
        if not os.path.exists(audio_wav_path):
            # print("Not exists", audio_wav_path)
            continue
        [audiowav, Fs] = torchaudio.load(audio_wav_path)
        audiowav = audiowav * (2 ** -23)
        
        n_samples = Fs/float(video_fps[i])
        starts=np.zeros(n_frames+1, dtype=int)
        ends=np.zeros(n_frames+1, dtype=int)
        starts[0]=0
        ends[0]=0
        for videoframe in range(1,n_frames+1):
            startemp=max(0,((videoframe-1)*(1.0/float(video_fps[i]))*Fs)-n_samples/2)
            starts[videoframe] = int(startemp)
            endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps[i]))*Fs)+n_samples/2))
            ends[videoframe] = int(endtemp)

        audioinfo = {
            'audiopath': audio_path,
            'video_id': video_names[i],
            'Fs' : Fs,
            'wav' : audiowav,
            'starts': starts,
            'ends' : ends
        }

        audiodata[video_names[i]] = audioinfo
    return audiodata

                             
def get_audio_feature(audioind, audiodata, clip_size, start_idx):
    len_snippet = clip_size
    max_audio_Fs = 22050
    min_video_fps = 10
    max_audio_win = int(max_audio_Fs / min_video_fps * 32)

    audioexcer  = torch.zeros(1,max_audio_win)
    valid = {}
    valid['audio']=0

    if audioind in audiodata:

        excerptstart = audiodata[audioind]['starts'][start_idx+1]
        if start_idx+len_snippet >= len(audiodata[audioind]['ends']):
            # print("Exceeds size", audioind)
            sys.stdout.flush()
            excerptend = audiodata[audioind]['ends'][-1]
        else:
            excerptend = audiodata[audioind]['ends'][start_idx+len_snippet] 
        try:
            valid['audio'] = audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
        except:
            pass
        audioexcer_tmp = audiodata[audioind]['wav'][:, excerptstart:excerptend+1]
        if (valid['audio']%2)==0:
            audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2))] = \
                torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
        else:
            audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2)+1)] = \
                torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp

    audio_feature = audioexcer.view(1,-1,1)
    return audio_feature


class SoundDatasetLoader(Dataset):
    def __init__(
        self,
        len_snippet,
        dataset_name='DIEM',
        split=1,
        mode='train',
        use_sound=False,
        use_vox=False,
        use_fixation=True,
                                            
        starts_per_video_range=(5, 6),
       
                                         
        default_skip_first_n=0,
        path_data=None,
    ):
 
        path_data = path_data or os.environ.get("A2_AUDIO_ROOT") or str(Path("datasets") / "Audio")

        self.path_data = path_data
        self.use_vox = use_vox
        self.use_sound = use_sound
        self.mode = mode
        self.len_snippet = len_snippet
        self.dataset_name = dataset_name
        self.split = split
        self.use_fixation = use_fixation

                                                  
        skip_map = {
            "AVAD": 5,
            "ETMD_av": 10,
            "SumMe": 10,
            "Coutrot_db1": 10,
            "Coutrot_db2": 10,
        }
        self.skip_first_n = int(skip_map.get(dataset_name, default_skip_first_n))

                      
        self.starts_per_video_range = starts_per_video_range
        self.epoch = 0

        self.img_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

                     
        if dataset_name == 'DIEM':
            list_file = f'DIEM_list_{mode}_fps.txt'
        else:
            list_file = f'{dataset_name}_list_{mode}_{split}_fps.txt'

        self.list_indata = []
                                    
        self.video_fps_dict = {}

        list_file_path = join(self.path_data, 'fold_lists', list_file)
        with open(list_file_path, 'r') as f:
            for line in f.readlines():
                                       
                parts = line.strip().split()
                name = parts[0]
                self.list_indata.append(name)
                
                              
                if len(parts) >= 3:
                    try:
                        self.video_fps_dict[name] = float(parts[2])
                    except ValueError:
                        self.video_fps_dict[name] = 30.0
                else:
                    self.video_fps_dict[name] = 30.0

        self.list_indata.sort()

                               
        self.video_num_frames = {}
        for v in self.list_indata:
            maps_dir = join(path_data, 'annotations', dataset_name, v, 'maps')
            if not os.path.isdir(maps_dir):
                continue
            frames = os.listdir(maps_dir)
            self.video_num_frames[v] = len(frames)

                     
        if self.mode == 'train':
            self._build_train_index()
        elif self.mode in ('test', 'val'):
            self._build_eval_index()
        else:
                                             
            self._build_eval_index()

        # audio
        max_audio_Fs = 22050
        min_video_fps = 10
        self.max_audio_win = int(max_audio_Fs / min_video_fps * 32)

        if use_sound or use_vox:
                                    
            audio_list_file = list_file
            if self.mode == 'val':
                audio_list_file = audio_list_file.replace('val', 'test')

            self.audiodata = make_dataset(
                join(self.path_data, 'fold_lists', audio_list_file),
                join(self.path_data, 'video_audio', self.dataset_name),
                join(self.path_data, 'annotations', self.dataset_name),
            )

    def check_frame(self, path):
        img = cv2.imread(path, 0)
        if img is None:
            return False
        return img.max() != 0

                                                     
    def _start_bounds(self, num_frames: int):
       
        min_start = self.skip_first_n
        max_start = num_frames - self.len_snippet
        return min_start, max_start

    def _build_train_index(self):
     
        self.train_index = []
        lo, hi = self.starts_per_video_range

        for v in self.list_indata:
            n = self.video_num_frames.get(v, 0)
            min_start, max_start = self._start_bounds(n)
            if max_start < min_start:
                continue                                 

            k = random.randint(lo, hi)
            for _ in range(k):
                self.train_index.append(v)

        random.shuffle(self.train_index)

    def _build_eval_index(self):
      
        self.eval_index = []
        step = 2 * self.len_snippet

        for v in self.list_indata:
            maps_dir = join(self.path_data, 'annotations', self.dataset_name, v, 'maps')
            if not os.path.isdir(maps_dir):
                continue

            frames = os.listdir(maps_dir)
            frames.sort()
            n = len(frames)

            min_start, max_start = self._start_bounds(n)
            if max_start < min_start:
                continue

                                     
            for i in range(min_start, max_start + 1, step):
                                                               
                probe_frame = i + self.len_snippet
                probe_path = join(maps_dir, 'eyeMap_%05d.jpg' % (probe_frame))
                if self.check_frame(probe_path):
                    self.eval_index.append((v, i))

                                             
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.mode == 'train':
            random.seed(epoch)
            np.random.seed(epoch)
            self._build_train_index()

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_index)
        return len(self.eval_index)

    def __getitem__(self, idx):
        if self.mode == "train":
            video_name = self.train_index[idx]
            num_frames = self.video_num_frames.get(video_name, 0)

            min_start, max_start = self._start_bounds(num_frames)
            if max_start < min_start:
                                                 
                start_idx = min_start
            else:
                                                                 
                while True:
                    start_idx = np.random.randint(min_start, max_start + 1)
                    probe_path = join(
                        self.path_data, 'annotations', self.dataset_name, video_name,
                        'maps', 'eyeMap_%05d.jpg' % (start_idx + self.len_snippet)
                    )
                    if self.check_frame(probe_path):
                        break
                    else:
                        print("No saliency defined in train dataset")
                        sys.stdout.flush()

        elif self.mode in ("test", "val"):
            (video_name, start_idx) = self.eval_index[idx]
                       
            if start_idx < self.skip_first_n:
                start_idx = self.skip_first_n
        else:
            (video_name, start_idx) = self.eval_index[idx]
            if start_idx < self.skip_first_n:
                start_idx = self.skip_first_n

        path_clip = join(self.path_data, 'video_frames', self.dataset_name, video_name)
        path_annt = join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps')

        clip_img = []
        clip_gt = []
        clip_fix = []      

        max_shift = 30

        for i in range(self.len_snippet):
            base_idx = start_idx + i + 1               
                                          
            if base_idx <= self.skip_first_n:
                base_idx = self.skip_first_n + 1

            frame_idx = base_idx
            shift = 0

            while True:
                gt_path = join(path_annt, 'eyeMap_%05d.jpg' % frame_idx)
                img_path = join(path_clip, 'img_%05d.jpg' % frame_idx)

                                   
                if (not os.path.exists(gt_path)) or (not os.path.exists(img_path)):
                    frame_idx = base_idx
                    gt_path = join(path_annt, 'eyeMap_%05d.jpg' % frame_idx)
                    img_path = join(path_clip, 'img_%05d.jpg' % frame_idx)
                    break

                gt_i = np.array(Image.open(gt_path).convert('L')).astype('float')

                if self.mode == "train":
                    gt_i = cv2.resize(gt_i, (448, 448))

                if np.max(gt_i) > 1.0:
                    gt_i = gt_i / 255.0

                               
                if gt_i.max() != 0:
                    break

                                                                    
                shift += 1
                frame_idx += 1

                if shift >= max_shift:
                    frame_idx = base_idx
                    gt_path = join(path_annt, 'eyeMap_%05d.jpg' % frame_idx)
                    img_path = join(path_clip, 'img_%05d.jpg' % frame_idx)

                    if os.path.exists(gt_path):
                        gt_i = np.array(Image.open(gt_path).convert('L')).astype('float')
                        if self.mode == "train":
                            gt_i = cv2.resize(gt_i, (448, 448))
                        if np.max(gt_i) > 1.0:
                            gt_i = gt_i / 255.0
                    else:
                        print('error')
                        gt_i = np.zeros((448, 448), dtype=np.float32)
                    break
                                        
            if self.use_fixation:
                fix_i = get_fixation_mat(self.path_data, self.dataset_name, video_name, frame_idx, key='eyeMap')
                if fix_i is None:
                                              
                    fix_i = np.zeros_like(gt_i, dtype=np.float32)
                else:
                                                
                    if self.mode == "train":
                        fix_i = cv2.resize(fix_i, (448, 448))

                                                      
                                                             
                    if np.max(fix_i) > 1.0:
                        fix_i = fix_i / 255.0

                clip_fix.append(torch.FloatTensor(fix_i))


            img = Image.open(img_path).convert('RGB')
            clip_img.append(self.img_transform(img))
            clip_gt.append(torch.FloatTensor(gt_i))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))               # [T,3,H,W]
        clip_gt  = torch.FloatTensor(torch.stack(clip_gt, dim=0)).unsqueeze(1)   # [T,1,H,W]
        if self.use_fixation:
            clip_fix = torch.FloatTensor(torch.stack(clip_fix, dim=0)).unsqueeze(1)  # [T,1,H,W]


        return_dict = {
            "image": clip_img,
            "saliency": clip_gt,
            "label": self.dataset_name}
        
        if self.use_fixation:
            return_dict["fixation"] = clip_fix

                           
        if self.use_sound:
                                         
            T_frames = clip_img.shape[0]
            
                                                             
            current_fps = self.video_fps_dict.get(video_name, 30.0)

                   
            audio_feature = get_frame_aligned_audio(
                audioind=video_name,
                audiodata=self.audiodata,
                clip_length=T_frames,
                start_idx=start_idx,
                video_fps=current_fps,
                target_sr=48000,                             
                context_sec=2.0          
            )
                                                        
            return_dict["audio"] = audio_feature.squeeze(1)

        return return_dict


def get_audio_feature_vox(audioind, audiodata, clip_size, start_idx):
    len_snippet = clip_size
    # max_audio_Fs = 22050
    # min_video_fps = 10
    max_audio_win = 48320

    audio_feature  = torch.zeros(max_audio_win)
    # valid = {}
    # valid['audio']=0

    if audioind in audiodata:

        excerptstart = audiodata[audioind]['starts'][start_idx+1]
        if start_idx+len_snippet >= len(audiodata[audioind]['ends']):
            # print("Exceeds size", audioind)
            sys.stdout.flush()
            excerptend = audiodata[audioind]['ends'][-1]
        else:
            excerptend = audiodata[audioind]['ends'][start_idx+len_snippet] 
        # try:
        #   valid['audio'] = audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
        # except:
        #   pass
        audio_feature_tmp = audiodata[audioind]['wav'][:, excerptstart:excerptend+1]

        if audio_feature_tmp.shape[1]<=audio_feature.shape[0]:
            audio_feature[:audio_feature_tmp.shape[1]] = audio_feature_tmp
        else:
            # print("Audio Length Bigger")
            audio_feature = audio_feature_tmp[0,:].copy()
    # print(audio_feature.shape)
    audio_feature = preprocess(audio_feature.numpy()).astype(np.float32)
    assert audio_feature.shape == (512,300), audio_feature.shape
    audio_feature=np.expand_dims(audio_feature, 2)
    return transforms.ToTensor()(audio_feature)
