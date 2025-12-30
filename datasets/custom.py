import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from Register import Registers
from datasets.base import *
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os
import torchvision.transforms.functional as TF
import numpy as np

# --- 1. CẤU HÌNH BẢNG MÀU CHUẨN ---
LC_CLASS_COLORS = torch.tensor([
    [0,   0,   0],        # Class 0: background/unknown
    [255, 0,   0],        # Class 1: building
    [133, 133, 133],      # Class 2: road
    [255, 0,   192],      # Class 3: parking
    [0,   180, 0],        # Class 4: tree
    [34,  139, 34],       # Class 5: forest
    [255, 193, 37],       # Class 6: cultivated land
    [128, 236, 104],      # Class 7: grass
    [0,   0,   255],      # Class 8: water
    [128, 0,   0],        # Class 9: barren
    [255, 255, 255],      # Class 10: others
], dtype=torch.float32)

_LC_RGB_TO_CLASS = {
    (255, 0, 0): 1,       # building
    (133, 133, 133): 2,   # road
    (255, 0, 192): 3,     # parking
    (0, 180, 0): 4,       # tree
    (34, 139, 34): 5,     # forest
    (255, 193, 37): 6,    # cultivated land
    (128, 236, 104): 7,   # grass
    (0, 0, 255): 8,       # water
    (128, 0, 0): 9,       # barren
    (255, 255, 255): 10,  # others
}


def encode_segmentation_rgb(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert LC RGB tensor -> label map.

    Requirements for correctness:
    - LC image must be resized with NEAREST (no blended colors).
    - LC must not be normalized before calling this.

    Args:
        image_tensor: (3,H,W) float tensor in [0,1] or [-1,1].
    Returns:
        (H,W) long tensor with labels in [0..10]. Unknown colors -> 0.
    """
    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected (3,H,W) tensor, got {tuple(image_tensor.shape)}")

    if image_tensor.min() < 0:
        image_denorm = (image_tensor + 1) * 127.5
    else:
        image_denorm = image_tensor * 255.0

    rgb = image_denorm.round().clamp(0, 255).to(torch.uint8)  # (3,H,W)
    rgb = rgb.permute(1, 2, 0).contiguous()  # (H,W,3)
    h, w, _ = rgb.shape

    label_map = torch.zeros((h, w), dtype=torch.long, device=image_tensor.device)
    for (r, g, b), cls in _LC_RGB_TO_CLASS.items():
        mask = (rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)
        label_map[mask] = cls
    return label_map

@Registers.datasets.register_with_name('SARtoOptical')
class CustomAlignedDataset_for_preprov7(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        
        dir_b = os.path.join(dataset_config.dataset_path, f'{stage}/B')
        dir_a = os.path.join(dataset_config.dataset_path, f'{stage}/A')
        dir_c = os.path.join(dataset_config.dataset_path, f'{stage}/C')

        image_paths_ori = get_image_paths_from_dir(dir_b)
        image_paths_cond = get_image_paths_from_dir(dir_a)
        image_paths_cond_c = get_image_paths_from_dir(dir_c)

        # Robust alignment by relative path (prevents silent mis-pairing if list orders differ)
        rel_b = {os.path.relpath(p, dir_b): p for p in image_paths_ori}
        rel_a = {os.path.relpath(p, dir_a): p for p in image_paths_cond}
        rel_c = {os.path.relpath(p, dir_c): p for p in image_paths_cond_c}
        common = sorted(set(rel_b.keys()) & set(rel_a.keys()) & set(rel_c.keys()))

        if len(common) == 0:
            raise RuntimeError(
                f"No aligned triplets found. Check folder structure under {dataset_config.dataset_path} for {stage}/A, {stage}/B, {stage}/C"
            )

        if not (len(common) == len(image_paths_ori) == len(image_paths_cond) == len(image_paths_cond_c)):
            print(
                f"[WARN] A/B/C counts mismatch or ordering differs for stage={stage}. "
                f"Using intersection={len(common)} (B={len(image_paths_ori)}, A={len(image_paths_cond)}, C={len(image_paths_cond_c)})."
            )

        image_paths_ori = [rel_b[k] for k in common]
        image_paths_cond = [rel_a[k] for k in common]
        image_paths_cond_c = [rel_c[k] for k in common]

        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond_c = ImagePathDataset(
            image_paths_cond_c,
            self.image_size,
            flip=self.flip,
            to_normal=False,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        # [SỬA LỖI Ở ĐÂY] Thêm [0] để lấy Tensor từ tuple (Tensor, Filename)
        item_ori = self.imgs_ori[i][0]       
        item_cond = self.imgs_cond[i][0]     
        item_cond_c = self.imgs_cond_c[i][0] # Ảnh RGB LC

        # Xử lý: RGB -> Label Index (0-10)
        label_lc = encode_segmentation_rgb(item_cond_c)
        
        return item_ori, item_cond, label_lc

# --- Các Class khác (Cũng cần sửa thêm [0] nếu dùng) ---
@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): 
        img = self.imgs[i][0] # Thêm [0]
        return img, img

@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)
    def __len__(self): return len(self.imgs_ori)
    def __getitem__(self, i): 
        return self.imgs_ori[i][0], self.imgs_cond[i][0] # Thêm [0]

# Các class Colorization/Inpainting giữ nguyên vì không dùng ImagePathDataset hoặc có logic riêng
@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)