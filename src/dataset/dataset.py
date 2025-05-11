import os
import random
import cv2
import numpy as np
import pandas as pd
import glob
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor
import torch.distributed as dist

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

class ValidDataset(Dataset): 
    """
    Dataset of videos, each video can be represented as:
        - an image of concatenated frames
        - '.mp4' or '.gif'
        - folder with all frames
    """
    def __init__(
            self,
            root_dir,
            sample_size=[512, 512], sample_stride=1, sample_n_frames=16,
            is_image=False, # stage 1인지 2인지 구분
            id_sampling=False, 
            pairs_list=None,
            data_list=None,
    ):
        self.root_dir = root_dir
        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size
        self.resize = transforms.Resize((sample_size[0], sample_size[1]))
        
        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[1], sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        self.is_image = is_image
        
        assert os.path.exists(os.path.join(root_dir, 'test'))
        self.root_dir = os.path.join(root_dir, 'test')
        test_videos = sorted(os.listdir(self.root_dir))
        
        self.pairs_list = pairs_list
        
        if self.is_image:
            self.videos = test_videos
        else:
            self.frame_sequences = []
      
            for video_name in test_videos:
                video_path = os.path.join(self.root_dir, video_name)
                frames = sorted(list(os.listdir(video_path)))
                num_frames = len(frames)
                
                num_sequences = num_frames // sample_n_frames
                for seq_idx in range(num_sequences):
                    start_frame = seq_idx * sample_n_frames
                    self.frame_sequences.append((video_name, start_frame))
       
        self.transform = None
        
    def __len__(self):
        if self.is_image:
            return len(self.videos)
        else:
            return len(self.frame_sequences)
    
    def get_batch_wo_pose(self, idx):
        if self.is_image:
            # Random frame selection for image pairs
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            frames = sorted(list(os.listdir(path)))
            video_length = len(frames)
            
            # Randomly select two frames
            frame_index = np.sort(np.random.choice(video_length, replace=True, size=2))
            src_idx = frame_index[0]
            
            # Read source frame
            src_path = os.path.join(path, frames[src_idx])
            src_img = cv2.imread(src_path)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            
            # Read target frame
            target_path = os.path.join(path, frames[frame_index[1]])
            target_img = cv2.imread(target_path)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            
            # Apply contrast normalization
            src_img = self.contrast_normalization(src_img)
            images_np = np.array([self.contrast_normalization(target_img)])
            
        else:
            # Sequential frame extraction
            video_name, start_frame = self.frame_sequences[idx]
            path = os.path.join(self.root_dir, video_name)
            frames = sorted(list(os.listdir(path)))
            
            # Get the frame indices for this sequence
            frame_indices = range(start_frame, start_frame + self.sample_n_frames)
        
            # Get source frame (first frame in sequence)
            src_idx = frame_indices[0]
            src_path = os.path.join(path, frames[src_idx])
            src_img = cv2.imread(src_path)
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            
            # Get target frames (rest of sequence)
            path_list = [os.path.join(path, frames[idx]) for idx in frame_indices]
      
            images = [cv2.imread(path) for path in path_list]
            images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images]
            
            # Apply contrast normalization
            src_img = self.contrast_normalization(src_img)
            images_np = np.array([self.contrast_normalization(img) for img in images])

        if self.is_image:
            images_np = images_np[0]
            
        name = str(os.path.basename(path) + '#' + str(src_idx))
        return src_img, images_np, name

    
    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx):
        src_img, tar_gt, name = self.get_batch_wo_pose(idx)
        sample = dict(
            src_img=src_img,
            tar_gt=tar_gt,
            name=name
            )
        
        return sample


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
        - an image of concatenated frames
        - '.mp4' or '.gif'
        - folder with all frames
    """
    def __init__(
            self,
            root_dir,
            sample_size=[512, 512], sample_stride=4, sample_n_frames=16, # stage 2에서 프레임 시퀀스 샘플링
            is_image=False, # stage 1인지 2인지 구분
            id_sampling=False, # id를 기준으로 클립 샘플링
            data_list=None  # CSV 파일 경로 추가
    ):
        self.root_dir = root_dir
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size
        self.resize = transforms.Resize((sample_size[0], sample_size[1]))
        
        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[1], sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        self.clip_image_processor = CLIPImageProcessor()
        
        self.is_image = is_image
        self.id_sampling = id_sampling
        self.data_list = data_list

        self.data = None
        
        assert os.path.exists(os.path.join(root_dir, 'train'))
        self.root_dir = os.path.join(self.root_dir, 'train')
        
        if data_list is not None:
            data = pd.read_csv(data_list)
            data.columns = ['video_folder', 'start_frame', 'end_frame']
            
            # id_sampling이 True인 경우, ID만 추출
            if id_sampling:
                # video_folder에서 ID 추출 (첫 번째 # 이전 부분)
                data['id'] = data['video_folder'].apply(lambda x: x.split('#')[0])
                # 고유 ID 목록 생성
                unique_ids = data['id'].unique()
                train_videos = sorted(list(unique_ids))
            else:
                # id_sampling이 False인 경우 video_folder 그대로 사용
                self.unique_video_folders = data['video_folder'].unique()
                train_videos = sorted(list(self.unique_video_folders))

            self.data = data
        else:
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(self.root_dir)}
                train_videos = sorted(list(train_videos))
            else:
                train_videos = sorted(os.listdir(self.root_dir))
        random.shuffle(train_videos) 
        
        self.videos = train_videos
        
        self.transform = None
        
    def __len__(self):
        return len(self.videos)
    
    def get_batch_wo_pose(self, idx):
        if self.data is not None:
            if self.id_sampling:
                video_id = self.videos[idx]
                video_entries = self.data[self.data['id'] == video_id]
                # 해당 ID를 가진 항목 중 하나를 랜덤하게 선택
                entry = video_entries.sample(1).iloc[0]
            else:
                video_name = self.videos[idx]
                video_entries = self.data[self.data['video_folder'] == video_name]
                entry = video_entries.iloc[0]
            
            path = os.path.join(self.root_dir, entry['video_folder'])
            start_frame = int(entry['start_frame'])
            end_frame = int(entry['end_frame'])
                
            frames = sorted(list(os.listdir(path)))

            if isinstance(frames[0], bytes):
                frames = [frame.decode('utf-8') for frame in frames]
            
            # 프레임 범위 제한
            if start_frame < len(frames) and end_frame < len(frames):
                frames = frames[start_frame:end_frame+1]
            
            video_length = len(frames)
            path_list = [os.path.join(path, frame) for frame in frames]
        else:
            if self.id_sampling:
                video_id = self.videos[idx]
                path = np.random.choice(glob.glob(os.path.join(self.root_dir, video_id + '*.mp4')))
            else:
                video_name = self.videos[idx]
                path = os.path.join(self.root_dir, video_name)

            frames = sorted(list(os.listdir(path)))
            
            if isinstance(frames[0], bytes):
                frames = [frame.decode('utf-8') for frame in frames]
        
            video_length = len(frames)
            path_list = [os.path.join(path, frame) for frame in frames] 

        tmp_sample_stride = self.sample_stride

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * tmp_sample_stride + 1)
            src_idx = np.sort(np.random.choice(video_length - clip_length, replace=True, size=1))[0]
            frame_index = np.linspace(src_idx, src_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else: 
            frame_index = np.sort(np.random.choice(video_length, replace=True, size=2)) 
            src_idx, frame_index = frame_index[0], frame_index[1:]
    
        src_img = cv2.imread(path_list[src_idx])
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        images = [cv2.imread(path_list[idx]) for idx in frame_index]
        images = [cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) for bgr_image in images]
        
        input_images = [src_img] + images
        if self.transform is not None and self.is_image:
            input_images = self.transform(input_images)
            
            src_img = input_images[0] 
            images = input_images[1:]
            
        src_img = self.contrast_normalization(src_img)
        images_np = np.array([self.contrast_normalization(img) for img in images])
        
        src_images_pil = Image.fromarray(src_img)
        src_image = self.clip_image_processor(images=src_images_pil, return_tensors="pt").pixel_values
        
        pixel_values_src = torch.from_numpy(src_img).permute(2, 0, 1).contiguous()
        pixel_values_src = pixel_values_src / 255.
        
        pixel_values_tar = torch.from_numpy(images_np).permute(0, 3, 1, 2).contiguous()
        pixel_values_tar = pixel_values_tar / 255.

        if self.is_image:
            pixel_values_tar = pixel_values_tar[0]
        
        return pixel_values_src, pixel_values_tar, src_image
    
    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx):
        pixel_values_src, pixel_values_tar, src_image = self.get_batch_wo_pose(idx)
        pixel_values_src = self.pixel_transforms(pixel_values_src)
        pixel_values_tar = self.pixel_transforms(pixel_values_tar)
        
        drop_image_embeds = 1 if random.random() < 0.1 else 0
        
        sample = dict(
            pixel_values_src=pixel_values_src, 
            pixel_values_tar=pixel_values_tar,
            src_image=src_image,
            drop_image_embeds=drop_image_embeds,
            )
        
        return sample
    
class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
    
    
class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)
        
        if pairs_list is None:
            # Extract IDs from video names
            videos = self.initial_dataset.videos
            video_ids = [name.split('#')[0] for name in videos]
            id_to_indices = {}
            
            # Group indices by IDs
            for idx, video_id in enumerate(video_ids):
                id_to_indices.setdefault(video_id, []).append(idx)

            # Create pairs with different IDs
            all_pairs = []
            id_keys = list(id_to_indices.keys())
            for i, id1 in enumerate(id_keys):
                for id2 in id_keys[i+1:]:
                    for idx1 in id_to_indices[id1]:
                        for idx2 in id_to_indices[id2]:
                            all_pairs.append((idx1, idx2))

            # Shuffle and select desired number of pairs
            np.random.shuffle(all_pairs)
            number_of_pairs = min(len(all_pairs), number_of_pairs)
            self.pairs = all_pairs[:number_of_pairs]
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}
    
        return {**first, **second}


def collate_fn(data): 
    pixel_values_src = torch.stack([example["pixel_values_src"] for example in data])
    pixel_values_tar = torch.stack([example["pixel_values_tar"] for example in data])
    src_image = torch.cat([example["src_image"] for example in data])
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)

    return {
        "pixel_values_src": pixel_values_src,
        "pixel_values_tar": pixel_values_tar,
        "src_image": src_image,
        "drop_image_embeds": drop_image_embeds,
    }


if __name__ == "__main__":
        train_dataset = FramesDataset(root_dir='./data/vox1_png', id_sampling=True, is_image=False, data_list='./data/filtered_dynamic_clips.csv')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,  
            shuffle=False,
            num_workers=0,  
            collate_fn=collate_fn,
        )

        # 첫 번째 배치 가져오기
        batch = next(iter(train_dataloader))

        print("DataLoader test completed successfully.")