import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from dataset_video import DHF1KDataset,Hollywood_UCFDataset,LEDOVDataset, DHF1KDataset_save
from data_process_uni import ImageDataset,ValDataset,FigrimDataset_Targets,FigrimDataset_Fillers
from torch.utils.data.distributed import DistributedSampler
import random
from collections import defaultdict
from torch.utils.data import Sampler
import pandas as pd
from dataset_audio import SoundDatasetLoader
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from pathlib import Path
import os


DEFAULT_DATA_ROOT = Path("datasets")


def resolve_data_root(data_root=None):
    return Path(data_root or os.environ.get("A2_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()


def data_path(data_root, *parts):
    return str(resolve_data_root(data_root).joinpath(*parts))

class AutoEpochDistributedSampler(Sampler):
 
    def __init__(self, dataset, shuffle=True, drop_last=False, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._iter_count = 0              

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
                              
        self._iter_count += 1
        g = torch.Generator()
        g.manual_seed(self.seed + self._iter_count)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

                              
        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]
        else:
            indices = indices[: self.total_size]

                        
        indices = indices[self.rank:self.total_size:self.num_replicas]

        if self.drop_last:
            indices = indices[: self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples



class GroupSampler(Sampler):
    def __init__(self, dataset, group_ids, batch_size, shuffle=True):
        self.dataset = dataset
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.shuffle = shuffle

        from collections import defaultdict
        import random
        self._random = random               

                                     
        self.group2indices = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            self.group2indices[gid].append(idx)

                  
        self.groups = list(self.group2indices.keys())

    def __iter__(self):
        all_batches = []

                             
        for gid in self.groups:
            indices = self.group2indices[gid]
            if self.shuffle:
                self._random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                all_batches.append(batch)

                                         
        if self.shuffle:
            self._random.shuffle(all_batches)

                        
        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for gid in self.groups:
            n = len(self.group2indices[gid])
            total_batches += (n + self.batch_size - 1) // self.batch_size
        return total_batches



def get_train_loader_image(batch_size, num_workers=8, data_root=None):
   
                     
                                                   
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448, 448)),                                        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

                                     
    train_dataset_Salicon = ImageDataset(
        ids_path=data_path(data_root, "salicon_256", "train_ids.csv"),
        stimuli_dir=data_path(data_root, "salicon_256", "stimuli", "train"),
        saliency_dir=data_path(data_root, "salicon_256", "saliency", "train"),
        fixation_dir=data_path(data_root, "salicon_256", "fixations", "train_edit"),
        dataset_name="Salicon",
        transform=train_transform,
    )

    train_dataset_OSIE = ImageDataset(
        ids_path=data_path(data_root, "OSIE_256", "train_id.csv"),
        stimuli_dir=data_path(data_root, "OSIE_256", "train", "train_stimuli"),
        saliency_dir=data_path(data_root, "OSIE_256", "train", "train_saliency"),
        fixation_dir=data_path(data_root, "OSIE_256", "train", "train_fixation"),
        dataset_name="OSIE",
        transform=train_transform,
    )

    train_dataset_CAT2000 = ImageDataset(
        ids_path=data_path(data_root, "CAT2000_256", "train_id.csv"),
        stimuli_dir=data_path(data_root, "CAT2000_256", "train", "train_stimuli"),
        saliency_dir=data_path(data_root, "CAT2000_256", "train", "train_saliency"),
        fixation_dir=data_path(data_root, "CAT2000_256", "train", "train_fixation"),
        dataset_name="CAT2000",
        transform=train_transform,
    )

    train_dataset_MIT1003 = ImageDataset(
        ids_path=data_path(data_root, "MIT1003_256", "train_id.csv"),
        stimuli_dir=data_path(data_root, "MIT1003_256", "train", "train_stimuli"),
        saliency_dir=data_path(data_root, "MIT1003_256", "train", "train_saliency"),
        fixation_dir=data_path(data_root, "MIT1003_256", "train", "train_fixation"),
        dataset_name="MIT1003",
        transform=train_transform,
    )

    train_dataset_SalEC = ImageDataset(
        ids_path=data_path(data_root, "SalEC", "train_ids.csv"),
        stimuli_dir=data_path(data_root, "SalEC", "train", "train_stimuli"),
        saliency_dir=data_path(data_root, "SalEC", "train", "train_saliency"),
        fixation_dir=data_path(data_root, "SalEC", "train", "train_fixation"),
        dataset_name="SalEC",
        transform=train_transform,
    )

    train_dataset_UI = ImageDataset(
        ids_path=data_path(data_root, "datasets_UI_256", "train_id.csv"),
        stimuli_dir=data_path(data_root, "datasets_UI_256", "train", "train_images"),
        saliency_dir=data_path(data_root, "datasets_UI_256", "train", "train_saliency"),
        fixation_dir=data_path(data_root, "datasets_UI_256", "train", "train_fixation"),
        dataset_name="UI",
        transform=train_transform,
    )
    
    train_dataset_fiwi = ImageDataset(
        ids_path=data_path(data_root, "fiwi_256", "train_id.csv"),
        stimuli_dir=data_path(data_root, "fiwi_256", "fiwi_train", "stimuli"),
        saliency_dir=data_path(data_root, "fiwi_256", "fiwi_train", "saliency"),
        fixation_dir=data_path(data_root, "fiwi_256", "fiwi_train", "fixations"),
        dataset_name="fiwi",
        transform=train_transform
    )
                     
                                        
    train_dataset_FIGRIM_Targets = FigrimDataset_Targets(
        ids_path=data_path(data_root, "Image", "FIGRIM", "Targets", "figrim_data.csv"),
        root_dir=data_path(data_root, "Image", "FIGRIM"),
        dataset_name="FIGRIM",
        transform=train_transform,                 
        hflip_p=0.5                            
    )
    train_dataset_FIGRIM_Fillers = FigrimDataset_Fillers(
        ids_path=data_path(data_root, "Image", "FIGRIM", "Fillers", "figrim_data.csv"),
        root_dir=data_path(data_root, "Image", "FIGRIM"),
        dataset_name="FIGRIM",
        transform=train_transform,                 
        hflip_p=0.5                            
    )

                   
    datasets = [
        # train_dataset_Salicon,
        # train_dataset_OSIE,
        train_dataset_CAT2000,
        # train_dataset_MIT1003,
        # train_dataset_SalEC,
        # train_dataset_UI,
        # train_dataset_fiwi,
        # train_dataset_FIGRIM_Targets, 
        # train_dataset_FIGRIM_Fillers
    ]
    
                          
    concat_dataset = ConcatDataset(datasets)

                                                 
    group_ids = []
    for gid, d in enumerate(datasets):
        group_ids.extend([gid] * len(d))

                        
    sampler = GroupSampler(
        dataset=concat_dataset,
        group_ids=group_ids,
        batch_size=batch_size,
        shuffle=True,
    )

    train_loader = DataLoader(
        concat_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, sampler


def get_val_loaders_image(batch_size, num_workers=8, data_root=None):

                                
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448, 448)),                          
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

                 
                              
    
    val_dataset_Salicon = ValDataset(
        ids_path=data_path(data_root, "salicon_256", "val_ids.csv"),
        stimuli_dir=data_path(data_root, "salicon_256", "stimuli", "val"),
        saliency_dir=data_path(data_root, "salicon_256", "saliency", "val"),
        fixation_dir=data_path(data_root, "salicon_256", "fixations", "val_edit"),
        dataset_name="Salicon",
        transform=val_transform,
    )

    val_dataset_OSIE = ValDataset(
        ids_path=data_path(data_root, "OSIE_256", "val_id.csv"),
        stimuli_dir=data_path(data_root, "OSIE_256", "val", "val_stimuli"),
        saliency_dir=data_path(data_root, "OSIE_256", "val", "val_saliency"),
        fixation_dir=data_path(data_root, "OSIE_256", "val", "val_fixation"),
        dataset_name="OSIE",
        transform=val_transform,
    )

    val_dataset_CAT2000 = ValDataset(
        ids_path=data_path(data_root, "CAT2000_256", "val_id.csv"),
        stimuli_dir=data_path(data_root, "CAT2000_256", "val", "val_stimuli"),
        saliency_dir=data_path(data_root, "CAT2000_256", "val", "val_saliency"),
        fixation_dir=data_path(data_root, "CAT2000_256", "val", "val_fixation"),
        dataset_name="CAT2000",
        transform=val_transform,
    )

    val_dataset_MIT1003 = ValDataset(
        ids_path=data_path(data_root, "MIT1003_256", "val_id.csv"),
        stimuli_dir=data_path(data_root, "MIT1003_256", "val", "val_stimuli"),
        saliency_dir=data_path(data_root, "MIT1003_256", "val", "val_saliency"),
        fixation_dir=data_path(data_root, "MIT1003_256", "val", "val_fixation"),
        dataset_name="MIT1003",
        transform=val_transform,
    )

    val_dataset_SalEC = ValDataset(
        ids_path=data_path(data_root, "SalEC", "val_ids.csv"),
        stimuli_dir=data_path(data_root, "SalEC", "val", "val_stimuli"),
        saliency_dir=data_path(data_root, "SalEC", "val", "val_saliency"),
        fixation_dir=data_path(data_root, "SalEC", "val", "val_fixation"),
        dataset_name="SalEC",
        transform=val_transform,
    )

    val_dataset_UI = ValDataset(
        ids_path=data_path(data_root, "datasets_UI_256", "val_id.csv"),
        stimuli_dir=data_path(data_root, "datasets_UI_256", "val", "val_images"),
        saliency_dir=data_path(data_root, "datasets_UI_256", "val", "val_saliency"),
        fixation_dir=data_path(data_root, "datasets_UI_256", "val", "val_fixation"),
        dataset_name="UI",
        transform=val_transform,
    )
    
    val_dataset_fiwi = ValDataset(
        ids_path=data_path(data_root, "fiwi_256", "val_id.csv"),
        stimuli_dir=data_path(data_root, "fiwi_256", "fiwi_val", "stimuli"),
        saliency_dir=data_path(data_root, "fiwi_256", "fiwi_val", "saliency"),
        fixation_dir=data_path(data_root, "fiwi_256", "fiwi_val", "fixations"),
        dataset_name="fiwi",
        transform=val_transform)


                         
    val_sampler_Salicon = DistributedSampler(val_dataset_Salicon, shuffle=False)
    val_sampler_OSIE = DistributedSampler(val_dataset_OSIE, shuffle=False)
    val_sampler_CAT2000 = DistributedSampler(val_dataset_CAT2000, shuffle=False)
    val_sampler_MIT1003 = DistributedSampler(val_dataset_MIT1003, shuffle=False)
    val_sampler_SalEC = DistributedSampler(val_dataset_SalEC, shuffle=False)
    val_sampler_UI = DistributedSampler(val_dataset_UI, shuffle=False)
    val_sampler_fiwi = DistributedSampler(val_dataset_fiwi, shuffle=False)
    
    loaders = {
        "Salicon": DataLoader(
            val_dataset_Salicon,
            batch_size=batch_size,
            sampler=val_sampler_Salicon,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "OSIE": DataLoader(
            val_dataset_OSIE,
            batch_size=batch_size,
            sampler=val_sampler_OSIE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "CAT2000": DataLoader(
            val_dataset_CAT2000,
            batch_size=batch_size,
            sampler=val_sampler_CAT2000,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "MIT1003": DataLoader(
            val_dataset_MIT1003,
            batch_size=batch_size,
            sampler=val_sampler_MIT1003,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "SalEC": DataLoader(
            val_dataset_SalEC,
            batch_size=batch_size,
            sampler=val_sampler_SalEC,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "UI": DataLoader(
            val_dataset_UI,
            batch_size=batch_size,
            sampler=val_sampler_UI,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
         "fiwi":DataLoader(
             val_dataset_fiwi, 
             batch_size=batch_size, sampler=val_sampler_fiwi, shuffle=False, num_workers=num_workers, pin_memory=False),

    }

    return loaders

def get_train_loader_video(batch_size, num_workers=8, data_root=None):
   
    train_path_data_DHF1K = data_path(data_root, "DHF1K", "DHF1K_train")
    train_dataset_DHF1K = DHF1KDataset(train_path_data_DHF1K, len_snippet=batch_size, epoch=0, mode="train")

    train_path_data_Hollywood = data_path(data_root, "Hollywood2_actions", "training")
  
    train_dataset_Hollywood = Hollywood_UCFDataset(
        train_path_data_Hollywood,
        len_snippet=batch_size,
        mode="train",
        samples_per_video=1,
        name="Hollywood",
    )

    train_path_data_UCF = data_path(data_root, "ucf-003", "training")
    train_dataset_UCF = Hollywood_UCFDataset(
        train_path_data_UCF,
        len_snippet=batch_size,
        mode="train",
        samples_per_video=1,
        name="UCF",
    )
    
    train_path_data_LEDOV = data_path(data_root, "LEDOV")
    excel_train_path = data_path(data_root, "LEDOV", "LEDOV_TrainValidVideoList.xlsx")
    train_df = pd.read_excel(excel_train_path, header=None)
    LEDOV_train_name_list = train_df[0].tolist()
    train_dataset_LEDOV = LEDOVDataset(train_path_data_LEDOV, len_snippet=batch_size, video_name_list = LEDOV_train_name_list, mode="train")


    audio_root = data_path(data_root, "Audio")
    train_dataset_diem = SoundDatasetLoader(batch_size, mode="train", dataset_name='DIEM', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    train_dataset_coutrout1 = SoundDatasetLoader(batch_size, mode="train", dataset_name='Coutrot_db1', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    train_dataset_coutrout2 = SoundDatasetLoader(batch_size, mode="train", dataset_name='Coutrot_db2', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    # train_dataset_avad = SoundDatasetLoader(batch_size, mode="train", dataset_name='AVAD', split=3, use_sound=True, use_vox=False)
    train_dataset_etmd = SoundDatasetLoader(batch_size, mode="train", dataset_name='ETMD_av', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    train_dataset_summe = SoundDatasetLoader(batch_size, mode="train", dataset_name='SumMe', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    
    full_dataset_list = [                                    
        train_dataset_LEDOV,
        train_dataset_diem,
        train_dataset_coutrout1,
        train_dataset_coutrout2,
        # train_dataset_avad,
        train_dataset_etmd,
        train_dataset_summe
    ]
    
    train_dataset = ConcatDataset(full_dataset_list)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, train_sampler
    

def get_val_loader_video(batch_size, num_workers=8, data_root=None):
   
    val_path_data_DHF1K = data_path(data_root, "DHF1K", "DHF1K_val")
    val_path_data_Hollywood = data_path(data_root, "Hollywood2_actions", "testing")
    val_path_data_UCF = data_path(data_root, "ucf-003", "testing")
    train_path_data_LEDOV = data_path(data_root, "LEDOV")
    excel_test_path = data_path(data_root, "LEDOV", "LEDOV_TestVideoList.xlsx")
    test_df = pd.read_excel(excel_test_path, header=None)
    LEDOV_test_name_list = test_df[0].tolist()
    val_dataset_LEDOV = LEDOVDataset(train_path_data_LEDOV, len_snippet=batch_size,video_name_list = LEDOV_test_name_list, mode="test")


    val_dataset_DHF1K = DHF1KDataset(val_path_data_DHF1K, len_snippet=batch_size, epoch=0, mode="val")
    val_dataset_UCF = Hollywood_UCFDataset(
        val_path_data_UCF,
        len_snippet=batch_size,
        mode="val",
        samples_per_video=1,
        name="UCF",
    )
    val_dataset_Hollywood = Hollywood_UCFDataset(
        val_path_data_Hollywood,
        len_snippet=batch_size,
        mode="val",
        samples_per_video=1,
        name="Hollywood",
    )
    
    # from torch.utils.data import Subset
    # val_dataset_DHF1K = Subset(val_dataset_DHF1K, range(300))                                                                          
    # val_dataset_LEDOV = Subset(val_dataset_LEDOV, range(300))

    
    audio_root = data_path(data_root, "Audio")
    val_dataset_diem = SoundDatasetLoader(batch_size, mode="test", dataset_name='DIEM', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    val_dataset_coutrout1 = SoundDatasetLoader(batch_size, mode="test", dataset_name='Coutrot_db1', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    val_dataset_coutrout2 = SoundDatasetLoader(batch_size, mode="test", dataset_name='Coutrot_db2', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    val_dataset_avad = SoundDatasetLoader(batch_size, mode="test", dataset_name='AVAD', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    val_dataset_etmd = SoundDatasetLoader(batch_size, mode="test", dataset_name='ETMD_av', split=3, use_sound=True, use_vox=False, path_data=audio_root)
    val_dataset_summe = SoundDatasetLoader(batch_size, mode="test", dataset_name='SumMe', split=3, use_sound=True, use_vox=False, path_data=audio_root)

    val_sampler_DHF1K = DistributedSampler(val_dataset_DHF1K, shuffle=False)
    val_sampler_UCF = DistributedSampler(val_dataset_UCF, shuffle=False)
    val_sampler_Hollywood = DistributedSampler(val_dataset_Hollywood, shuffle=False)
    val_sampler_LEDOV = DistributedSampler(val_dataset_LEDOV, shuffle=False)
    


    val_loader_DHF1K = DataLoader(
        val_dataset_DHF1K,
        batch_size=1,
        sampler=val_sampler_DHF1K,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader_UCF = DataLoader(
        val_dataset_UCF,
        batch_size=1,
        sampler=val_sampler_UCF,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader_Hollywood = DataLoader(
        val_dataset_Hollywood,
        batch_size=1,
        sampler=val_sampler_Hollywood,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader_LEDOV = DataLoader(
        val_dataset_LEDOV,
        batch_size=1,
        sampler=val_sampler_LEDOV,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    val_sampler_diem = DistributedSampler(val_dataset_diem, shuffle=False)
    val_sampler_coutrout1 = DistributedSampler(val_dataset_coutrout1, shuffle=False)
    val_sampler_coutrout2 = DistributedSampler(val_dataset_coutrout2, shuffle=False)
    val_sampler_avad = DistributedSampler(val_dataset_avad, shuffle=False)
    val_sampler_etmd = DistributedSampler(val_dataset_etmd, shuffle=False)
    val_sampler_summe = DistributedSampler(val_dataset_summe, shuffle=False)
    
    test_dataset_DHF1K  = DHF1KDataset_save(
    path_data=data_path(data_root, "DHF1K", "DHF1K_val", "video"),
    old_img_root=data_path(data_root, "DHF1K", "DHF1K_val", "video"),
    new_img_root=data_path(data_root, "DHF1K", "DHF1K_save"),
    copy_to_new_root=False)
    test_sampler_DHF1K = DistributedSampler(test_dataset_DHF1K, shuffle=False)
    test_loader_DHF1K = DataLoader(
        test_dataset_DHF1K,
        batch_size=1,
        sampler=test_sampler_DHF1K,
        num_workers=num_workers,
        pin_memory=False)


    loaders = {
        # "DHF1K": test_loader_DHF1K,
        "DHF1K": val_loader_DHF1K,
        "UCF": val_loader_UCF,
        "Hollywood": val_loader_Hollywood,
        "LEDOV": val_loader_LEDOV,
        "DIEM": DataLoader(val_dataset_diem, batch_size=1, sampler=val_sampler_diem, num_workers=num_workers, pin_memory=False),
        "Coutrot_db1": DataLoader(val_dataset_coutrout1, batch_size=1, sampler=val_sampler_coutrout1, num_workers=num_workers, pin_memory=False),
        "Coutrot_db2": DataLoader(val_dataset_coutrout2, batch_size=1, sampler=val_sampler_coutrout2, num_workers=num_workers, pin_memory=False),
        "AVAD": DataLoader(val_dataset_avad, batch_size=1, sampler=val_sampler_avad, num_workers=num_workers, pin_memory=False),
        "ETMD_av": DataLoader(val_dataset_etmd, batch_size=1, sampler=val_sampler_etmd, num_workers=num_workers, pin_memory=False),
        "SumMe": DataLoader(val_dataset_summe, batch_size=1, sampler=val_sampler_summe, num_workers=num_workers, pin_memory=False),
    }

    return loaders
