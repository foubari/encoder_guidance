import random
from pathlib import Path
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


DATASET_REGISTRY = {}

def register_dataset(name):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


# Get the directory of the current file
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

@register_dataset("night2day")
class Night2Day(Dataset):
    """
    Returns a 3-tuple: (day_tensor, night_tensor, label)
    where label = 1 → real paired (x_t, y_t)
          label = 0 → fake independent pair
    """
    _exts = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, 
                 real_root=os.path.join(data_dir, 'night2day_data/separated_night_day'), 
                 fake_day_dir=os.path.join(data_dir, 'night2day_data/samples/day/1_2_4/tr_stp_70000_stp1000/2025_07_20_22_40'),
                 fake_night_dir=os.path.join(data_dir, 'night2day_data/samples/night/1_2_4/tr_stp_70000_stp1000/2025_07_21_06_55'),
                 image_size=64, p_real=0.5, split="train"):
        super().__init__()

        real_root = Path(real_root)
        self.day_dir = real_root / "day" / split
        self.night_dir = real_root / "night" / split

        ids_day = {p.stem.rsplit("_day", 1)[0] for p in self.day_dir.iterdir()
                   if p.suffix.lower() in self._exts}
        ids_night = {p.stem.rsplit("_night", 1)[0] for p in self.night_dir.iterdir()
                     if p.suffix.lower() in self._exts}
        self.real_ids = sorted(ids_day & ids_night)
        if not self.real_ids:
            raise RuntimeError("No paired real images found!")

        print(fake_day_dir, fake_night_dir)
        self.fake_day_paths = sorted([p for p in Path(fake_day_dir).iterdir()
                                      if p.suffix.lower() in self._exts])
        self.fake_night_paths = sorted([p for p in Path(fake_night_dir).iterdir()
                                        if p.suffix.lower() in self._exts])
        
        if not self.fake_day_paths or not self.fake_night_paths:
            raise RuntimeError("No fake images found in the supplied folders")

        self.tf = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        self.p_real = p_real

    def __len__(self):
        return len(self.real_ids)

    def _load_img(self, path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        if random.random() < self.p_real:
            id_ = self.real_ids[idx % len(self.real_ids)]
            day = self._load_img(self.day_dir / f"{id_}_day.jpg")
            night = self._load_img(self.night_dir / f"{id_}_night.jpg")
            label = 1
        else:
            day = self._load_img(random.choice(self.fake_day_paths))
            night = self._load_img(random.choice(self.fake_night_paths))
            label = 0

        return day, night, torch.tensor(label, dtype=torch.float32)

def get_data(dataset_name,
             batch_size=32,
             num_workers=4,
             pin_memory=True,
             shuffle=True,
             **dataset_kwargs):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")
    
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(**dataset_kwargs)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)
