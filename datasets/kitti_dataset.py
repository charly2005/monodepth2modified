import os
import numpy as np
import PIL as pil
from PIL import Image
import torch
from torch.utils import data

class KITTIDataset(data.Dataset):

    # in calib file the intrinsics matrix is the first 9 value of p2 for the left camera, and the first 9 of p3 for right
    # image 2 is left
    # image 3 is right

    TRAIN_SEQUENCES = ['00','01','02','03','04','05','06','07','08']
    VAL_SEQUENCES = ['09', '10']
        
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # idx, seq name, start, end
    DATA_MAP = {
        '00': ['2011_10_03_drive_0027_sync', '000000', '004540'],
        '01': ['2011_10_03_drive_0042_sync', '000000', '001100'],
        '02': ['2011_10_03_drive_0034_sync', '000000', '004660'],
        '03': ['2011_09_26_drive_0067_sync', '000000', '000800'],
        '04': ['2011_09_30_drive_0016_sync', '000000', '000270'],
        '05': ['2011_09_30_drive_0018_sync', '000000', '002760'],
        '06': ['2011_09_30_drive_0020_sync', '000000', '001100'],
        '07': ['2011_09_30_drive_0027_sync', '000000', '001100'],
        '08': ['2011_09_30_drive_0028_sync', '001100', '005170'],
        '09': ['2011_09_30_drive_0033_sync', '000000', '001590'],
        '10': ['2011_09_30_drive_0034_sync', '000000', '001200'],
    }

    def __init__(self, root_dir, eigen_dir, mode = 'train'):
        # KITTI/data_odometry_color/dataset/
        super().__init__()
        self.root_dir = root_dir

        self.mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
 
        self.mode = mode
        if self.mode == 'train':
            self.sequences = self.TRAIN_SEQUENCES
        elif self.mode == 'val':
            self.sequences = self.VAL_SEQUENCES

        self.eigen_split = self._load_eigen_split(eigen_dir)

        self.image_files, self.intrinsics, self.samples= self._get_files(root_dir, self.eigen_split)


        

    def _parse_projection_mat(self, lines, key):
        for line in lines:
            if line.startswith(key):

                data = np.array(line.strip().split()[1:], dtype=np.float32)
                return data.reshape(3, 4)

    def _load_intrinsics(self, path):
        with open(path, 'r') as f:
            calib = f.readlines()
        
        P2 = self._parse_projection_mat(calib, 'P2:')
        P3 = self._parse_projection_mat(calib, 'P3:')
        K_left = torch.from_numpy(P2[:3, :3]).float()
        K_right = torch.from_numpy(P3[:3, :3]).float()
        
        return K_left, K_right
        

    def _get_files(self, root_dir, eigen_split):

        # image_files[seq idx][camera_idx 0 = left 1 = right][frame_idx] = filename
        image_files = []

        # intrisics[seq idx] = [left_K, right_K]
        intrinsics = []

        # samples[idx] = [seq idx, camera_idx, frame_idx]
        samples = []
        print(root_dir)
        # first 10 are for training/val
        for seq in self.sequences:
            seq_path = os.path.join(root_dir, 'sequences', seq)

            # image files
            left_camera = os.path.join(seq_path, 'image_2')
            right_camera = os.path.join(seq_path, 'image_3')

            left_camera_files = []
            drive_path = self.DATA_MAP[seq][0]
            start_frame = int(self.DATA_MAP[seq][1])

            for i, f in enumerate(sorted(os.listdir(left_camera))[1:-1]):

                raw_idx = start_frame + i + 1 # we skip the first one cuz im lazy, technically left_camera_files should include the first and last images
                if drive_path not in eigen_split:
                    left_camera_files.append(f)
                elif raw_idx not in eigen_split[drive_path]:
                    left_camera_files.append(f)
            
            right_camera_files = []

            for i, f in enumerate(sorted(os.listdir(right_camera))[1:-1]):
                right_camera_files.append(f)

            left_paths = [os.path.join(left_camera, f) for f in left_camera_files]
            right_paths = [os.path.join(right_camera, f) for f in right_camera_files]

            image_files.append([left_paths, right_paths])

            seq_idx = len(image_files) -1

            for i in range(1, len(left_camera_files) - 1):
                curr_id = int(os.path.splitext(left_camera_files[i])[0])
                prev_id = int(os.path.splitext(left_camera_files[i-1])[0])
                next_id = int(os.path.splitext(left_camera_files[i+1])[0])

                if (curr_id == prev_id + 1) and (curr_id == next_id - 1):
                    samples.append([seq_idx, 0, i])

            for i in range(1, len(right_camera_files) - 1):
                curr_id = int(os.path.splitext(right_camera_files[i])[0])
                prev_id = int(os.path.splitext(right_camera_files[i-1])[0])
                next_id = int(os.path.splitext(right_camera_files[i+1])[0])

                if (curr_id == prev_id + 1) and (curr_id == next_id - 1):
                    samples.append([seq_idx, 1, i])

            # intrinsics loading
            calib = os.path.join(seq_path, 'calib.txt')
            K_left, K_right = self._load_intrinsics(calib)
            
            intrinsics.append([K_left, K_right])

        return image_files, intrinsics, samples

    def __len__(self):
        return len(self.samples)

    def _load_eigen_split(self, eigen_split_dir):
        # store drive info, idx
        eigen_split = {}
        with open(eigen_split_dir, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # ex: 2011_10_03_drive_0027_sync
                drive_path = parts[0].split('/')[1]
                # ex: 0000002725
                idx = int(parts[1])
                
                if drive_path not in eigen_split:
                    eigen_split[drive_path] = {}

                eigen_split[drive_path][idx] = 1
        return eigen_split

    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _process_image(self, image):

        img_resized = image.resize((768, 768), resample=pil.Image.BILINEAR)

        img_np = np.array(img_resized, dtype=np.float32) / 255.0
 
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        mean = self.mean
        std = self.std
        
        img_tensor.sub_(mean).div_(std)
        
        return img_tensor

    def __getitem__(self, idx):

        seq_idx, camera_idx, frame_idx = self.samples[idx]

        path = self.image_files[seq_idx][camera_idx][frame_idx]
        l_path = self.image_files[seq_idx][camera_idx][frame_idx-1]
        r_path = self.image_files[seq_idx][camera_idx][frame_idx+1]

        image = self._pil_loader(path)
        h_o, w_o = image.size
        l_image = self._pil_loader(l_path)
        r_image = self._pil_loader(r_path)
        

        
        t = self._process_image(image)
        l_t = self._process_image(l_image)
        r_t = self._process_image(r_image)

        K = self.intrinsics[seq_idx][camera_idx]

        return {
            't': t,
            'l_t': l_t,
            'r_t': r_t,
            'K': K,
            'w_o':w_o,
            'h_o':h_o,
        }


        