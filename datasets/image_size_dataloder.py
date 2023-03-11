from pathlib import Path
import random
import json
import itertools
import copy
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, \
    SequentialSampler
from torchvision import transforms
import numpy as np
import cv2
import PIL
from datasets.dataset import default_data_dir
from utils.utils import normalize_tensor

config_path = Path('datasets')

class MIT1003Dataset(Dataset):

    source = 'MIT1003'
    n_train_val_images = 1003
    dynamic = False

    def __init__(self, phase='train', subset=None,
                 preproc_cfg=None, n_x_val=10, x_val_step=0, x_val_seed=27):
        self.phase = phase
        self.train = phase == 'train'
        self.subset = subset

        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)

        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed

        # Cross-validation split
        n_images = self.n_train_val_images
        if x_val_step is None:
            self.samples = np.arange(0, n_images)
        else:
            print(f"X-Val step: {x_val_step}")
            assert(self.x_val_step < self.n_x_val)
            samples = np.arange(0, n_images)
            if self.x_val_seed > 0:
                np.random.seed(self.x_val_seed)
                np.random.shuffle(samples)
            val_start = int(len(samples) / self.n_x_val * self.x_val_step)
            val_end = int(len(samples) / self.n_x_val * (self.x_val_step + 1))
            samples = samples.tolist()
            if not self.train:
                self.samples = samples[val_start:val_end]
            else:
                del samples[val_start:val_end]
                self.samples = samples

        self.all_image_files, self.size_dict = self.load_data()
        if self.subset is not None:
            self.samples = self.samples[:int(len(self.samples) * subset)]
        # For compatibility with video datasets
        self.n_images_dict = {sample: 1 for sample in self.samples}
        self.target_size_dict = {
            img_idx: self.size_dict[img_idx]['target_size']
            for img_idx in self.samples}
        self.n_samples = len(self.samples)
        self.frame_modulo = 1

    def get_map(self, img_idx):
        map_file = self.fix_dir / self.all_image_files[img_idx]['map']
        map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        assert(map is not None)
        return map

    def get_img(self, img_idx):
        img_file = self.img_dir / self.all_image_files[img_idx]['img']
        img = cv2.imread(str(img_file))
        assert(img is not None)
        return np.ascontiguousarray(img[:, :, ::-1])

    def get_fixation_map(self, img_idx):
        fix_map_file = self.fix_dir / self.all_image_files[img_idx]['pts']
        fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        assert(fix_map is not None)
        return fix_map

    @property
    def dir(self):
        return Path(default_data_dir / 'MIT1003')

    @property
    def fix_dir(self):
        return self.dir / 'ALLFIXATIONMAPS' / 'ALLFIXATIONMAPS'

    @property
    def img_dir(self):
        return self.dir / 'ALLSTIMULI' / 'ALLSTIMULI'

    def get_out_size_eval(self, img_size):
        ar = img_size[0] / img_size[1]

        min_prod = 100
        max_prod = 120
        ar_array = []
        size_array = []
        for n1 in range(7, 14):
            for n2 in range(7, 14):
                if min_prod <= n1 * n2 <= max_prod:
                    this_ar = n1 / n2
                    this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                    ar_array.append(this_ar_ratio)
                    size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)

        return out_size

    def get_out_size_train(self, img_size):
        selection = (8, 13), (9, 13), (9, 12), (12, 9), (13, 9)  # 待选高宽比
        ar = img_size[0] / img_size[1]  # 原图高宽比
        ar_array = []
        size_array = []
        for n1, n2 in selection:
            this_ar = n1 / n2
            this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))   # 原图高宽比与改待选高宽比的差距
            ar_array.append(this_ar_ratio)
            size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx] # 选择较小差距的作为高宽比
        out_size = tuple(r * 32 for r in bn_size)

        return out_size

    def load_data(self):

        all_image_files = []
        for img_file in sorted(self.img_dir.glob("*.jpeg")):
            all_image_files.append({
                'img': img_file.name,
                'map': img_file.stem + "_fixMap.jpg",
                'pts': img_file.stem + "_fixPts.jpg",
            })
            assert((self.fix_dir / all_image_files[-1]['map']).exists())
            assert((self.fix_dir / all_image_files[-1]['pts']).exists())

        size_dict_file = config_path / "img_size_dict.json"
        if size_dict_file.exists():
            with open(size_dict_file, 'r') as f:
                size_dict = json.load(f)
                size_dict = {int(img_idx): val for
                                  img_idx, val in size_dict.items()}
        else:
            size_dict = {}
            for img_idx in range(self.n_train_val_images):
                img = cv2.imread(
                    str(self.img_dir / all_image_files[img_idx]['img']))
                size_dict[img_idx] = {'img_size': img.shape[:2]}
            with open(size_dict_file, 'w') as f:
                json.dump(size_dict, f)

        for img_idx in self.samples:
            img_size = size_dict[img_idx]['img_size']
            if self.phase in ('train', 'valid'):
                out_size = self.get_out_size_train(img_size)
            else:
                out_size = self.get_out_size_eval(img_size)
            if self.phase in ('train', 'valid'):
                target_size = tuple(sz * 2 for sz in out_size)
            else:
                target_size = img_size

            size_dict[img_idx].update({
                'out_size': out_size, 'target_size': target_size})

        return all_image_files, size_dict

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, out_size=None, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        if data in ('img', 'sal'):
            transformations.append(transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS))
        else:
            transformations.append(transforms.Resize(
                out_size, interpolation=PIL.Image.NEAREST))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_idx):
        img = self.get_img(img_idx)
        out_size = self.size_dict[img_idx]['out_size']
        target_size = self.target_size_dict[img_idx]
        img = self.preprocess(img, out_size=out_size, data='img')
        if self.phase == 'test':
            return [1], img, target_size

        sal = self.get_map(img_idx)
        sal = self.preprocess(sal, target_size, data='sal')
        fix = self.get_fixation_map(img_idx)
        fix = self.preprocess(fix, target_size, data='fix')

        return [1], img, sal, fix, target_size

    def __getitem__(self, item):
        img_idx = self.samples[item]
        return self.get_data(img_idx)

class ImgSizeBatchSampler:

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        assert(isinstance(dataset, MIT1003Dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        out_size_array = [
            dataset.size_dict[img_idx]['out_size']
            for img_idx in dataset.samples]
        self.out_size_set = sorted(list(set(out_size_array)))
        self.sample_idx_dict = {
            out_size: [] for out_size in self.out_size_set}
        for sample_idx, img_idx in enumerate(dataset.samples):
            self.sample_idx_dict[dataset.size_dict[img_idx]['out_size']].append(
                sample_idx)

        self.len = 0
        self.n_batches_dict = {}
        for out_size, sample_idx_array in self.sample_idx_dict.items():
            this_n_batches = len(sample_idx_array) // self.batch_size
            self.len += this_n_batches
            self.n_batches_dict[out_size] = this_n_batches

    def __iter__(self):
        batch_array = list(itertools.chain.from_iterable(
            [out_size for _ in range(n_batches)]
            for out_size, n_batches in self.n_batches_dict.items()))
        if not self.shuffle:
            random.seed(27)
        random.shuffle(batch_array)

        this_sample_idx_dict = copy.deepcopy(self.sample_idx_dict)
        for sample_idx_array in this_sample_idx_dict.values():
            random.shuffle(sample_idx_array)
        for out_size in batch_array:
            this_indices = this_sample_idx_dict[out_size][:self.batch_size]
            del this_sample_idx_dict[out_size][:self.batch_size]
            yield this_indices

    def __len__(self):
        return self.len


class ImgSizeDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 **kwargs):
        if batch_size == 1:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            batch_sampler = ImgSizeBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle,
                drop_last=drop_last)
        super().__init__(dataset, batch_sampler=batch_sampler, **kwargs)


def getMIT1003Dataloder(phase, batch_size, shuffle=False, drop_last=False,):
    dataset = MIT1003Dataset(phase=phase)
    dataloder = ImgSizeDataLoader(dataset, batch_size, shuffle, drop_last)
    return dataloder