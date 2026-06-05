import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

        
TARGET_SIZE = 448
import pandas as pd
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random
                      
TARGET_SIZE = 448

    
import os
import cv2
import numpy as np
import pandas as pd
import torch
import scipy.io as sio             
from PIL import Image
from torch.utils.data import Dataset
import random




class ImageDataset(Dataset):
    def __init__(self, ids_path, stimuli_dir, saliency_dir, fixation_dir,
                 dataset_name, transform=None, hflip_p=0.5):
        self.ids = pd.read_csv(ids_path)
        self.stimuli_dir = stimuli_dir
        self.saliency_dir = saliency_dir
        self.fixation_dir = fixation_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.hflip_p = hflip_p               

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
                         
        do_flip = random.random() < self.hflip_p

        # ---------- Load image ----------
        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')

        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            image = self.transform(image)

        # ---------- Load saliency map ----------
        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        saliency = saliency.resize((448, 448), Image.BILINEAR)
        saliency = np.array(saliency, dtype=np.float32) / 255.0

        if do_flip:
            saliency = np.fliplr(saliency)

        saliency = torch.from_numpy(saliency.copy()).unsqueeze(0)

        # ---------- Load fixation map ----------
        fmap_path = self.fixation_dir + self.ids.iloc[idx, 2]
        fixation = Image.open(fmap_path).convert('L')
        fixation = fixation.resize((448, 448), Image.NEAREST)
        fixation = np.array(fixation, dtype=np.float32) / 255.0

        if do_flip:
            fixation = np.fliplr(fixation)

        fixation = torch.from_numpy(fixation.copy()).unsqueeze(0)

        sample = {
            'image': image,
            'saliency': saliency,
            'fixation': fixation,
            'label': self.dataset_name
        }
        return sample

    
    
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

                      
TARGET_SIZE = 448

class ValDataset(Dataset):
    def __init__(self, ids_path, stimuli_dir, saliency_dir, fixation_dir, dataset_name, transform=None):
        self.ids = pd.read_csv(ids_path)
        self.stimuli_dir = stimuli_dir
        self.saliency_dir = saliency_dir
        self.fixation_dir = fixation_dir
        self.dataset_name = dataset_name        
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # ---------- Load image ----------
        im_filename = self.ids.iloc[idx, 0]
        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        
        image = Image.open(im_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # ---------- Load saliency map ----------
        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        saliency = saliency.resize((448, 448), Image.BILINEAR)
        saliency = np.array(saliency, dtype=np.float32) / 255.0
        saliency = torch.from_numpy(saliency).unsqueeze(0)

        # ---------- Load fixation map ----------
        fmap_path = self.fixation_dir + self.ids.iloc[idx, 2]
        fixation = Image.open(fmap_path).convert('L')
        fixation = fixation.resize((448, 448), Image.NEAREST)
        fixation = np.array(fixation, dtype=np.float32) / 255.0
        fixation = torch.from_numpy(fixation).unsqueeze(0)

        # ---------- Label ----------
                                        
        label = self.dataset_name

        sample = {
            'image': image,
            'saliency': saliency,
            'fixation': fixation,
            'label': label,
            'im_path': im_filename
        }
        return sample

def preprocess_img(img_dir, channels=3):
    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    image_org = img
    shape_r = TARGET_SIZE
    shape_c = TARGET_SIZE

    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
                    
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2)
                   :((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
                    
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2)
                   :((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded, image_org


def postprocess_img(pred, org_dir):
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2)
                   :((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2)
                   :((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img



import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import random



class FigrimDataset_Targets(Dataset):
    def __init__(self, ids_path, root_dir, dataset_name="FIGRIM", transform=None, hflip_p=0.5):
  
        self.ids = pd.read_csv(ids_path)
        self.root_dir = root_dir
        
                      
                                   
        self.stimuli_dir = os.path.join(self.root_dir, 'Targets', 'Targets') 
        
                                    
                                                                        
        self.gt_dir = os.path.join(self.root_dir, 'Targets', 'FIXATIONMAPS') 
        
        self.dataset_name = dataset_name
        self.transform = transform
        self.hflip_p = hflip_p

    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self, idx):
                                                                             
        im_rel_path = self.ids.iloc[idx, 0] 
        im_path = os.path.join(self.stimuli_dir, im_rel_path)
        
                    
        if not os.path.exists(im_path):
            im_path = im_path.replace('\\', '/')
        
                      
        try:
            image = Image.open(im_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Cannot load image: {im_path}")
  

        W, H = image.size 

                                                                          
        gt_rel_path = self.ids.iloc[idx, 1] 
        gt_path = os.path.join(self.gt_dir, gt_rel_path)
        
                 
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace('\\', '/')
                                                       
            if not os.path.exists(gt_path) and gt_path.endswith('.jpg'):
                 if os.path.exists(gt_path.replace('.jpg', '.png')):
                     gt_path = gt_path.replace('.jpg', '.png')

                                
        try:
            gt_image = Image.open(gt_path).convert('L')
        except Exception as e:
            # print(f"[Warning] Cannot load GT: {gt_path}, using blank map.")
            gt_image = Image.new('L', (W, H))

        # ================= 3. Preprocessing (Resize & Flip) =================
                
        do_flip = random.random() < self.hflip_p
        
                                
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
                   
        image_resized = image.resize((448, 448), Image.BILINEAR)
        
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            image_tensor = torch.from_numpy(np.array(image_resized)).permute(2,0,1).float()/255.

                             
        if do_flip:
            gt_image = gt_image.transpose(Image.FLIP_LEFT_RIGHT)
        
                                             
        smap = gt_image.resize((448, 448), Image.BILINEAR)
        smap = np.array(smap, dtype=np.float32) / 255.0
        
                             
                                                          
                                   
        fmap = gt_image.resize((448, 448), Image.NEAREST)
        fmap = np.array(fmap, dtype=np.float32) / 255.0

                                                       
        smap = torch.from_numpy(smap).unsqueeze(0)
        fmap = torch.from_numpy(fmap).unsqueeze(0)

        sample = {
            'image': image_tensor,
            'saliency': smap,
            'fixation': fmap,                                            
            'label': self.dataset_name
        }
        return sample
    
    

class FigrimDataset_Fillers(Dataset):
    def __init__(self, ids_path, root_dir, dataset_name="FIGRIM", transform=None, hflip_p=0.5):
        self.ids = pd.read_csv(ids_path)
        self.root_dir = root_dir
        
                      
                                   
        self.stimuli_dir = os.path.join(self.root_dir, 'Fillers', 'Fillers') 
        
                                    
                                                                        
        self.gt_dir = os.path.join(self.root_dir, 'Fillers', 'FIXATIONMAPS') 
        
        self.dataset_name = dataset_name
        self.transform = transform
        self.hflip_p = hflip_p

    def __len__(self):
        return len(self.ids)
    
                                         

    def __getitem__(self, idx):
                                                                             
        im_rel_path = self.ids.iloc[idx, 0] 
        im_path = os.path.join(self.stimuli_dir, im_rel_path)
        
                    
        if not os.path.exists(im_path):
            im_path = im_path.replace('\\', '/')
        
                      
        try:
            image = Image.open(im_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Cannot load image: {im_path}")
                         
            image = Image.new('RGB', (448, 448))

        W, H = image.size 

                                                                          
        gt_rel_path = self.ids.iloc[idx, 1] 
        gt_path = os.path.join(self.gt_dir, gt_rel_path)
        
                 
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace('\\', '/')
                                                       
            if not os.path.exists(gt_path) and gt_path.endswith('.jpg'):
                 if os.path.exists(gt_path.replace('.jpg', '.png')):
                     gt_path = gt_path.replace('.jpg', '.png')

                                
        try:
            gt_image = Image.open(gt_path).convert('L')
        except Exception as e:
            # print(f"[Warning] Cannot load GT: {gt_path}, using blank map.")
            gt_image = Image.new('L', (W, H))

        # ================= 3. Preprocessing (Resize & Flip) =================
                
        do_flip = random.random() < self.hflip_p
        
                                
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
                   
        image_resized = image.resize((448, 448), Image.BILINEAR)
        
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            image_tensor = torch.from_numpy(np.array(image_resized)).permute(2,0,1).float()/255.

                             
        if do_flip:
            gt_image = gt_image.transpose(Image.FLIP_LEFT_RIGHT)
        
                                             
        smap = gt_image.resize((448, 448), Image.BILINEAR)
        smap = np.array(smap, dtype=np.float32) / 255.0
        
                             
                                                          
                                    
        fmap = gt_image.resize((448, 448), Image.NEAREST)
        fmap = np.array(fmap, dtype=np.float32) / 255.0

                                                       
        smap = torch.from_numpy(smap).unsqueeze(0)
        fmap = torch.from_numpy(fmap).unsqueeze(0)

        sample = {
            'image': image_tensor,
            'saliency': smap,
            'fixation': fmap,                                            
            'label': self.dataset_name
        }
        return sample
