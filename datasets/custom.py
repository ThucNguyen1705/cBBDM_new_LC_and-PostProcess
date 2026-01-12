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


def extract_lc_edges(label_map: torch.Tensor, min_length: int = 20, dilate_size: int = 1) -> torch.Tensor:
    """
    Extract boundary edges from LC label map based on CLASS TRANSITIONS.
    
    This method focuses on boundaries between different LC classes (color changes),
    which is more appropriate for semantic segmentation maps than gradient-based methods.
    
    Algorithm:
    1. Detect pixels where neighboring pixels have different class labels
    2. Create binary boundary map
    3. Remove small isolated components (noise)
    4. Optional dilation for visibility
    
    Args:
        label_map: (H, W) long tensor with class labels 0..10
        min_length: Minimum contour length to keep (removes tiny fragments)
        dilate_size: Dilation kernel size (0 = no dilation)
    
    Returns:
        edge_map: (1, H, W) float tensor in [0, 1], where 1 = boundary
    """
    # Convert to numpy
    label_np = label_map.cpu().numpy().astype(np.int32)
    h, w = label_np.shape
    
    # Step 1: Detect class transitions (boundaries between different classes)
    # Check horizontal neighbors (left-right)
    boundary_h = np.zeros((h, w), dtype=np.uint8)
    boundary_h[:, 1:] = (label_np[:, 1:] != label_np[:, :-1]).astype(np.uint8)
    boundary_h[:, :-1] |= (label_np[:, :-1] != label_np[:, 1:]).astype(np.uint8)
    
    # Check vertical neighbors (up-down)
    boundary_v = np.zeros((h, w), dtype=np.uint8)
    boundary_v[1:, :] = (label_np[1:, :] != label_np[:-1, :]).astype(np.uint8)
    boundary_v[:-1, :] |= (label_np[:-1, :] != label_np[1:, :]).astype(np.uint8)
    
    # Check diagonal neighbors (for smoother boundaries)
    boundary_d1 = np.zeros((h, w), dtype=np.uint8)
    boundary_d1[1:, 1:] = (label_np[1:, 1:] != label_np[:-1, :-1]).astype(np.uint8)
    
    boundary_d2 = np.zeros((h, w), dtype=np.uint8)
    boundary_d2[1:, :-1] = (label_np[1:, :-1] != label_np[:-1, 1:]).astype(np.uint8)
    
    # Combine all boundaries
    boundary = (boundary_h | boundary_v | boundary_d1 | boundary_d2).astype(np.uint8)
    
    # Step 2: Remove small connected components (keep only long boundaries)
    if min_length > 0:
        # Find contours and filter by length
        contours, _ = cv2.findContours(boundary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        boundary_filtered = np.zeros_like(boundary)
        for contour in contours:
            if len(contour) >= min_length:
                cv2.drawContours(boundary_filtered, [contour], -1, 1, thickness=1)
        boundary = boundary_filtered
    
    # Step 3: Optional dilation for better visibility
    if dilate_size > 0:
        kernel = np.ones((dilate_size * 2 + 1, dilate_size * 2 + 1), np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)
    
    # Convert back to tensor: (1, H, W) float in [0, 1]
    edge_tensor = torch.from_numpy(boundary.astype(np.float32)).unsqueeze(0)
    
    return edge_tensor

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
        
        # Extract edge map from LC label map based on class boundaries
        # min_length=50: Chỉ giữ các đường biên dài (loại bỏ chi tiết nhỏ vụn)
        # dilate_size=0: Không làm dày thêm (giữ đường mảnh)
        # edge_map: (1, H, W) float tensor in [0, 1]
        edge_map = extract_lc_edges(label_lc, min_length=50, dilate_size=0)
        
        return item_ori, item_cond, label_lc, edge_map

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


# =====================================================
# Phase 2 Refinement Dataset with Pre-generated Coarse RGB
# =====================================================
@Registers.datasets.register_with_name('Phase2Refinement')
class Phase2RefinementDataset(Dataset):
    """
    Dataset for Phase 2 Texture Refinement training.
    
    Loads pre-generated coarse RGB from disk instead of generating on-the-fly.
    This dramatically speeds up training.
    
    Expected folder structure:
        {dataset_path}/{split}/
            ├── A/          # SAR images
            ├── B/          # Optical GT
            ├── C/          # LC label maps
            └── coarse_rgb/ # Pre-generated from Phase 1 (run pre_generate_coarse.py)
    
    Returns:
        (optical_gt, sar, lc_label, edge_map, coarse_rgb)
    """
    
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        
        # Paths
        dir_a = os.path.join(dataset_config.dataset_path, f'{stage}/A')  # SAR
        dir_b = os.path.join(dataset_config.dataset_path, f'{stage}/B')  # Optical GT
        dir_c = os.path.join(dataset_config.dataset_path, f'{stage}/C')  # LC
        dir_coarse = os.path.join(dataset_config.dataset_path, f'{stage}/coarse_rgb')  # Pre-generated
        
        # Check coarse_rgb exists
        if not os.path.exists(dir_coarse):
            raise RuntimeError(
                f"Coarse RGB folder not found: {dir_coarse}\n"
                f"Please run: python pre_generate_coarse.py --config configs/Template-Refinement.yaml --split {stage}"
            )
        
        # Get image paths
        image_paths_ori = get_image_paths_from_dir(dir_b)
        image_paths_sar = get_image_paths_from_dir(dir_a)
        image_paths_lc = get_image_paths_from_dir(dir_c)
        image_paths_coarse = get_image_paths_from_dir(dir_coarse)
        
        # Align by relative path
        rel_b = {os.path.relpath(p, dir_b): p for p in image_paths_ori}
        rel_a = {os.path.relpath(p, dir_a): p for p in image_paths_sar}
        rel_c = {os.path.relpath(p, dir_c): p for p in image_paths_lc}
        
        # For coarse_rgb, use index-based alignment (generated in order)
        common = sorted(set(rel_b.keys()) & set(rel_a.keys()) & set(rel_c.keys()))
        
        if len(common) == 0:
            raise RuntimeError(f"No aligned samples found for stage={stage}")
        
        # Check coarse_rgb count matches
        if len(image_paths_coarse) != len(common):
            raise RuntimeError(
                f"Coarse RGB count mismatch: {len(image_paths_coarse)} vs {len(common)} samples.\n"
                f"Please regenerate coarse RGB with: python pre_generate_coarse.py --split {stage}"
            )
        
        self.image_paths_ori = [rel_b[k] for k in common]
        self.image_paths_sar = [rel_a[k] for k in common]
        self.image_paths_lc = [rel_c[k] for k in common]
        self.image_paths_coarse = sorted(image_paths_coarse)  # Sorted by filename (000000.png, 000001.png, ...)
        
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.transform_lc = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        print(f"[Phase2RefinementDataset] Loaded {len(self)} samples for {stage}")
    
    def __len__(self):
        return len(self.image_paths_ori)
    
    def __getitem__(self, i):
        # Load images
        img_ori = Image.open(self.image_paths_ori[i]).convert('RGB')
        img_sar = Image.open(self.image_paths_sar[i]).convert('RGB')
        img_lc = Image.open(self.image_paths_lc[i]).convert('RGB')
        img_coarse = Image.open(self.image_paths_coarse[i]).convert('RGB')
        
        # Random horizontal flip (apply same flip to all)
        if self.flip and random.random() > 0.5:
            img_ori = TF.hflip(img_ori)
            img_sar = TF.hflip(img_sar)
            img_lc = TF.hflip(img_lc)
            img_coarse = TF.hflip(img_coarse)
        
        # Transform
        item_ori = self.transform(img_ori)
        item_sar = self.transform(img_sar)
        item_lc = self.transform_lc(img_lc)
        item_coarse = self.transform(img_coarse)
        
        # Normalize to [-1, 1]
        if self.to_normal:
            item_ori = (item_ori - 0.5) * 2.0
            item_sar = (item_sar - 0.5) * 2.0
            item_coarse = (item_coarse - 0.5) * 2.0
        
        # Encode LC to label map
        label_lc = encode_segmentation_rgb(item_lc)
        
        # Extract edge map
        edge_map = extract_lc_edges(label_lc, min_length=50, dilate_size=0)
        
        return item_ori, item_sar, label_lc, edge_map, item_coarse