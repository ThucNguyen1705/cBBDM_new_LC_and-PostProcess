import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
import torchvision.utils as vutils

@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        print(f'optimizer = {str(config.model.BB.optimizer.optimizer)}')
        if config.model.BB.optimizer.optimizer=='AdamW':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   **vars(config.model.BB.lr_scheduler))
        
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                mode='min',
                                                                verbose=True,
                                                                threshold_mode='rel',
                                                                **vars(config.model.BB.lr_scheduler)
)
            print(f'actual scheduler optimizer = {str(scheduler.optimizer)}')
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name = self._unpack_batch(batch)
            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            
            # [SỬA] Đảm bảo x_cond_lc là Long và có shape đúng
            if x_cond_lc is not None:
                x_cond_lc = x_cond_lc.to(self.config.training.device[0])
                if x_cond_lc.dim() == 4: x_cond_lc = x_cond_lc.squeeze(1)
                x_cond_lc = x_cond_lc.long()
            else:
                x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long() # Mock label map
            
            # Handle edge_map
            if x_cond_edge is not None:
                x_cond_edge = x_cond_edge.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_sar_latent = self.net.encode(x_cond_sar, cond=True, normalize=False)
            
            # [SỬA] Gọi get_cond_stage_context thay vì encode trực tiếp cho LC
            x_cond_lc_latent = self.net.get_cond_stage_context(x_cond_lc, x_cond_sar, x_cond_edge)

            # concat latent (Lưu ý: Nếu dimension không khớp thì cần chỉnh lại logic này)
            # Nếu dùng MARM, output là 128 kênh, SAR latent là 3 kênh. Không concat được kiểu cũ.
            # Tạm thời chỉ dùng SAR latent để tính mean/std cho BBDM gốc (nếu cần)
            x_cond_latent = x_cond_sar_latent 

            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name = self._unpack_batch(batch)
            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            if x_cond_lc is not None:
                x_cond_lc = x_cond_lc.to(self.config.training.device[0])
                if x_cond_lc.dim() == 4: x_cond_lc = x_cond_lc.squeeze(1)
                x_cond_lc = x_cond_lc.long()
            else:
                x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
            
            # Handle edge_map
            if x_cond_edge is not None:
                x_cond_edge = x_cond_edge.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_sar_latent = self.net.encode(x_cond_sar, cond=True, normalize=False)
            
            x_cond_latent = x_cond_sar_latent # Giữ logic đơn giản cho phần latent mean

            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def _unpack_batch(self, batch):
        x = x_name = x_cond_sar = x_cond_sar_name = x_cond_lc = x_cond_edge = None

        if not (isinstance(batch, (list, tuple)) and len(batch) >= 3):
            # Fallback cũ (2 elements)
            if isinstance(batch, (list, tuple)):
                first = batch[0]
                second = batch[1] if len(batch) > 1 else None
            else:
                x = batch; x_name = None
                return x, x_name, None, None, None, None
            
            if isinstance(first, (list, tuple)): x = first[0]; x_name = first[1]
            else: x = first; x_name = None
            
            if isinstance(second, (list, tuple)): x_cond_sar = second[0]; x_cond_name = second[1]
            else: x_cond_sar = second; x_cond_name = None
            
            return x, x_name, x_cond_sar, None, None, x_cond_name

        # Logic mới: 3 hoặc 4 phần tử (optical, sar, lc_label, [edge_map])
        if isinstance(batch[0], (list, tuple)): x = batch[0][0]; x_name = batch[0][1]
        else: x = batch[0]; x_name = None

        if isinstance(batch[1], (list, tuple)): x_cond_sar = batch[1][0]; x_cond_sar_name = batch[1][1]
        else: x_cond_sar = batch[1]; x_cond_sar_name = None

        if isinstance(batch[2], (list, tuple)): x_cond_lc = batch[2][0]
        else: x_cond_lc = batch[2]
        
        # Check for 4th element: edge_map
        if len(batch) >= 4:
            if isinstance(batch[3], (list, tuple)): x_cond_edge = batch[3][0]
            else: x_cond_edge = batch[3]
            
        return x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_sar_name
    
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name = self._unpack_batch(batch)

        x = x.to(self.config.training.device[0])
        x_cond_sar = x_cond_sar.to(self.config.training.device[0])
        if x_cond_lc is None:
            x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
        else:
            x_cond_lc = x_cond_lc.to(self.config.training.device[0])
        
        # Handle edge_map
        if x_cond_edge is not None:
            x_cond_edge = x_cond_edge.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond_sar, x_cond_lc, x_cond_edge)

        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if isinstance(additional_info, dict):
                if 'recloss_noise' in additional_info:
                    self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
                if 'recloss_xy' in additional_info:
                    self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
                if 'sem_loss' in additional_info:
                    self.writer.add_scalar(f'sem_loss/{stage}', additional_info['sem_loss'], step)
        return loss

    # === [HÀM MỚI] GIẢI MÃ LABEL -> RGB ===
    def decode_seg_map_for_vis(self, label_batch):
        """
        Chuyển đổi Tensor Nhãn (Long) -> Tensor Ảnh RGB (Float) để vẽ.
        Input: (B, H, W) hoặc (B, 1, H, W)
        Output: (B, 3, H, W) trong khoảng [-1, 1]
        """
        # Bảng màu chuẩn (khớp với datasets/custom.py) cho 11 classes (0..10)
        # 0: background/unknown
        # 1..10: theo LC_CLASS_COLORS
        colors = torch.tensor([
            [0,   0,   0],        # 0: background/unknown
            [255, 0,   0],        # 1: building
            [133, 133, 133],      # 2: road
            [255, 0,   192],      # 3: parking
            [0,   180, 0],        # 4: tree
            [34,  139, 34],       # 5: forest
            [255, 193, 37],       # 6: cultivated land
            [128, 236, 104],      # 7: grass
            [0,   0,   255],      # 8: water
            [128, 0,   0],        # 9: barren
            [255, 255, 255],      # 10: others
        ], device=label_batch.device, dtype=torch.float32) / 255.0
        
        # Xử lý kích thước: (B, 1, H, W) -> (B, H, W)
        if label_batch.dim() == 4: 
            label_batch = label_batch.squeeze(1)
        
        # Đảm bảo kiểu Long và kẹp giá trị an toàn
        label_batch = label_batch.long()
        label_batch = torch.clamp(label_batch, min=0, max=10)

        # Mapping: (B, H, W) -> (B, H, W, 3)
        rgb = F.embedding(label_batch, colors)
        
        # Permute: (B, 3, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
        
        # Chuyển về khoảng [-1, 1] để phù hợp với hàm get_image_grid
        rgb = (rgb - 0.5) * 2.0
        return rgb

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name = self._unpack_batch(batch)

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond_sar = x_cond_sar[0:batch_size].to(self.config.training.device[0])
        if x_cond_lc is None:
            x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
        else:
            x_cond_lc = x_cond_lc[0:batch_size].to(self.config.training.device[0])
        
        # Handle edge_map
        if x_cond_edge is not None:
            x_cond_edge = x_cond_edge[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        if self.config.testing.sample_num > 1:
            samples, one_step_samples = net.sample(x_cond_sar, x_cond_lc, x_cond_edge, clip_denoised=self.config.testing.clip_denoised, sample_mid_step=True)
            self.save_images(samples, reverse_sample_path, grid_size, save_interval=200, writer_tag=f'{stage}_sample' if stage != 'test' else None)
            self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=200, writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
            sample = samples[-1]
        else:
            sample = net.sample(x_cond_sar, x_cond_lc, x_cond_edge, clip_denoised=self.config.testing.clip_denoised)
            sample = sample.to('cpu')

        # 1. Vẽ ảnh dự đoán (Sample)
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        # 2. Vẽ ảnh SAR (Condition 1)
        image_grid_sar = get_image_grid(x_cond_sar.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid_sar)
        im.save(os.path.join(sample_path, 'condition_sar.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition_sar', image_grid_sar, self.global_step, dataformats='HWC')

        # 3. Vẽ ảnh LC (Condition 2) - Label map -> RGB visualization
        if x_cond_lc.dtype == torch.long or x_cond_lc.dtype == torch.int:
            vis_lc = self.decode_seg_map_for_vis(x_cond_lc) # -> RGB Tensor
            image_grid_lc = get_image_grid(vis_lc.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        else:
            image_grid_lc = get_image_grid(x_cond_lc.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
            
        im = Image.fromarray(image_grid_lc)
        im.save(os.path.join(sample_path, 'condition_lc.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition_lc', image_grid_lc, self.global_step, dataformats='HWC')

        # 4. Vẽ ảnh Edge Map (Condition 3) - nếu có
        if x_cond_edge is not None:
            # Edge map là (B, 1, H, W) float in [0,1] -> expand to 3 channels for visualization
            edge_vis = x_cond_edge.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            edge_vis = (edge_vis - 0.5) * 2.0  # Convert to [-1, 1] for get_image_grid
            image_grid_edge = get_image_grid(edge_vis.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
            im = Image.fromarray(image_grid_edge)
            im.save(os.path.join(sample_path, 'condition_edge.png'))
            if stage != 'test':
                self.writer.add_image(f'{stage}_condition_edge', image_grid_edge, self.global_step, dataformats='HWC')

        # 5. Vẽ ảnh thật (Ground Truth)
        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        condition_sar_path = make_dir(os.path.join(sample_path, f'condition_sar'))
        condition_lc_path = make_dir(os.path.join(sample_path, f'condition_lc'))
        condition_edge_path = make_dir(os.path.join(sample_path, f'condition_edge'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        
        for test_batch in pbar:
            x, x_name, x_cond_sar, x_cond_lc, x_cond_edge, x_cond_name = self._unpack_batch(test_batch)
            # Fix name unpacking
            x_cond_sar_name = x_cond_name

            x = x.to(self.config.training.device[0])
            x_cond_sar = x_cond_sar.to(self.config.training.device[0])
            if x_cond_lc is None:
                x_cond_lc = torch.zeros_like(x_cond_sar)[:, 0, :, :].long()
            else:
                x_cond_lc = x_cond_lc.to(self.config.training.device[0])
            
            # Handle edge_map
            if x_cond_edge is not None:
                x_cond_edge = x_cond_edge.to(self.config.training.device[0])

            # [SỬA] Chuẩn bị bản visual của LC để lưu ảnh
            if x_cond_lc.dtype == torch.long:
                x_cond_lc_vis = self.decode_seg_map_for_vis(x_cond_lc)
            else:
                x_cond_lc_vis = x_cond_lc

            for j in range(sample_num):
                sample = net.sample(x_cond_sar, x_cond_lc, x_cond_edge, clip_denoised=False)
                for i in range(batch_size):
                    # Lưu các ảnh điều kiện và kết quả
                    condition_sar = x_cond_sar[i].detach().clone()
                    
                    # Dùng bản VISUAL (RGB) để lưu file ảnh LC
                    condition_lc = x_cond_lc_vis[i].detach().clone()
                    
                    gt = x[i]
                    result = sample[i]
                    
                    if j == 0:
                        name_sar = x_cond_sar_name[i] if (x_cond_sar_name is not None and len(x_cond_sar_name)>i) else f"sample_{i}"
                        save_single_image(condition_sar, condition_sar_path, f'{name_sar}.png', to_normal=to_normal)
                        
                        name_lc = f"lc_{i}"
                        save_single_image(condition_lc, condition_lc_path, f'{name_lc}.png', to_normal=to_normal)
                        
                        # Save edge map if available
                        if x_cond_edge is not None:
                            edge_vis = x_cond_edge[i].repeat(3, 1, 1)  # (3, H, W)
                            edge_vis = (edge_vis - 0.5) * 2.0  # Convert to [-1, 1]
                            save_single_image(edge_vis, condition_edge_path, f'edge_{i}.png', to_normal=to_normal)
                        
                        gt_name = x_name[i] if (x_name is not None and len(x_name)>i) else f"gt_{i}"
                        save_single_image(gt, gt_path, f'{gt_name}.png', to_normal=to_normal)
                        
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.png', to_normal=to_normal)