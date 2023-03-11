import abc
import random
from pathlib import Path
import os
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import scipy.io
from utils import utils
from utils.utils import preprocess_size_img, preprocess_size_fixmap

default_data_dir = Path("/data/qmengyu/01-Datasets/02-Saliency-Dataset/")


class SaliencyDataset(Dataset):

    def __init__(self, out_size, source, phase,  val_rito=0.2):

        self.source = source
        self.phase = phase  # train, val, test, all: all represents the fusion of training data and validation data

        self.train = phase == 'train'
        self.out_size = out_size
        self.val_rito = val_rito
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        self.dir = default_data_dir / self.source.upper()

    def get_img(self, img_file):
        img = cv2.imread(str(img_file)) # H, W, C , bgr
        assert (img is not None)
        img_size = [img.shape[0], img.shape[1]]
        return np.ascontiguousarray(img[:, :, ::-1]), img_size


    def get_salmap(self, map_file):
        salmap = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        assert (salmap is not None)
        return salmap

    def get_fixation_map(self, fix_map_file):
        fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        assert (fix_map is not None)
        return fix_map


    @abc.abstractmethod
    def file_name_to_path(self, file_name):
        pass

    def get_other_maps(self, ):
        """Sample reference maps for s-AUC"""
        n_aucs_maps = 10
        while True:
            this_map = np.zeros(self.out_size)
            sample_files = random.sample(
                self.samples, n_aucs_maps)

            for file_name in sample_files:
                paths = self.file_name_to_path(file_name)
                this_this_map = self.get_fixation_map(paths[2])
                this_this_map = cv2.resize(
                    this_this_map, (self.out_size[1], self.out_size[0]),
                    cv2.INTER_NEAREST
                )

                this_map += this_this_map

            this_map = np.clip(this_map, 0, 1)
            yield this_map

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, data='img'):
        channels = 3 if data == 'img' else 1
        if data in ('img', 'sal'):
            img = preprocess_size_img(img, self.out_size[0], self.out_size[1], channels=channels)

        if data in ('fix',):
            img = preprocess_size_fixmap(img, self.out_size[0], self.out_size[1])

        transformations = [transforms.ToPILImage(), transforms.ToTensor()]

        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def __getitem__(self, index):

        file_name = self.samples[index]
        paths = self.file_name_to_path(file_name)

        img, ori_size = self.get_img(paths[0])
        img = self.preprocess(img, data='img')
        if self.phase == 'test':
            return {"imgs": img,
                    "ori_sizes": torch.tensor(ori_size),
                    "file_names": file_name}

        sal = self.get_salmap(paths[1])
        sal = self.preprocess(sal, data='sal')

        fix = self.get_fixation_map(paths[2])
        fix = self.preprocess(fix, data='fix')

        return {"imgs": img,
                "sals": sal,
                "fix_maps":  fix,
                "ori_sizes":  torch.tensor(ori_size),
                "file_names":  file_name}

class SALICONDataset(SaliencyDataset):
    def __init__(self, out_size, val_rito, source="salicon", phase='train'):
        super(SALICONDataset, self).__init__(out_size, source, phase,  val_rito=val_rito)

        assert phase in ('train', 'val')
        self.out_size = out_size
        self.phase = phase
        train_samples = [file.stem for file in (self.dir / 'images' / 'train').glob('*.jpg')]
        val_samples = [file.stem for file in (self.dir / 'images' / 'val').glob('*.jpg')]

        self.samples = train_samples if self.phase == "train" else val_samples
        self.images_dir = self.dir / 'images' / self.phase
        self.sals_dir = self.dir / "maps" / self.phase
        self.fixs_dir = self.dir / "fixations" / self.phase

    def from_raw_fixations(self, raw_fix_file):
        print(f"SALICON-------{raw_fix_file}-----load_from_raw_fixations")
        fix_data = scipy.io.loadmat(raw_fix_file)
        fixations_array = [gaze[2] for gaze in fix_data['gaze'][:, 0]]
        res = fix_data['resolution'].tolist()[0]
        fix_map = np.zeros(res, dtype=np.uint8)
        for subject_fixations in fixations_array:
            fix_map[subject_fixations[:, 1] - 1, subject_fixations[:, 0] - 1] \
                = 255
        return fix_map

    def get_fixation_map(self, fix_map_file):
        if fix_map_file.exists():
            fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        else:
            fix_map = self.from_raw_fixations(str(fix_map_file)[:-4] + '.mat')
            cv2.imwrite(str(fix_map_file), fix_map)

        return fix_map

    def file_name_to_path(self, file_name):
        img_path = self.images_dir / (file_name + '.jpg')
        sal_path = self.sals_dir / (file_name + '.png')
        fix_path = self.fixs_dir / (file_name + '.png')

        return img_path, sal_path, fix_path


class MIT1003Dataset(SaliencyDataset):
    def __init__(self, out_size, val_rito, source="mit1003", phase='train'):
        super(MIT1003Dataset, self).__init__(out_size, source, phase,  val_rito=val_rito)

        assert phase in ('train', 'val', 'test')
        samples = [file.stem for file in (self.dir / 'ALLSTIMULI' / 'ALLSTIMULI').glob('*.jpeg')]
        train_samples = samples[:int(len(samples) * (1 - self.val_rito))]
        val_samples = samples[int(len(samples) * (1 - self.val_rito)):]
        self.images_dir = self.dir / 'ALLSTIMULI' / 'ALLSTIMULI'
        self.sals_dir = self.dir / "ALLFIXATIONMAPS" / "ALLFIXATIONMAPS"
        self.fixs_dir = self.dir / "ALLFIXATIONMAPS" / "ALLFIXATIONMAPS"
        if self.phase == "train":
            self.samples = train_samples
        elif self.phase == "val":
            self.samples = val_samples
        else:
            self.samples = samples

    def file_name_to_path(self, file_name):
        img_path = self.images_dir / (file_name + '.jpeg')
        sal_path = self.sals_dir / (file_name + '_fixMap.jpg')
        fix_path = self.fixs_dir / (file_name + '_fixPts.jpg')

        return img_path, sal_path, fix_path


class MIT300Dataset(SaliencyDataset):
    def __init__(self, out_size, val_rito, source="mit300", phase='test'):
        super(MIT300Dataset, self).__init__(out_size, source, phase,  val_rito=val_rito)

        assert phase == 'test'
        test_samples = [file.stem for file in (self.dir / 'BenchmarkIMAGES' / 'BenchmarkIMAGES').glob('*.jpg')]

        self.images_dir = self.dir / 'BenchmarkIMAGES' / 'BenchmarkIMAGES'
        self.samples = test_samples if self.phase == "test" else None

    def file_name_to_path(self, file_name):
        img_path = self.images_dir / (file_name + '.jpg')
        return img_path, None, None


# not finish
class CAT2000Dataset(Dataset):
    def __init__(self, out_size, val_rito,  source="cat2000", phase='train'):
        super(CAT2000Dataset, self).__init__(source, phase, out_size, val_rito)

        assert phase in ('train', 'val', 'test')
        samples = [os.path.join(file.parent.stem, file.stem)
                   for file in (self.dir / 'trainSet' / 'Stimuli').rglob('^[0-9]*.jpg')]
        train_samples = samples[:int(len(samples) * (1 - self.val_rito))]
        val_samples = samples[int(len(samples) * (1 - self.val_rito)):]
        test_samples = [os.path.join(file.parent.stem, file.stem)
                        for file in (self.dir / 'testSet' / 'Stimuli').rglob('^[0-9]*.jpg')]

        self.images_dir = self.dir / 'trainSet' / 'Stimuli' if self.phase != "test" \
            else self.dir / 'testSet' / 'Stimuli'

        self.sals_dir = self.dir / 'trainSet' / 'FIXATIONMAPS' if self.phase != "test" \
            else self.dir / 'testSet' / 'FIXATIONMAPS'

        # self.fixs_dir = self.dir / "ALLFIXATIONMAPS" / "ALLFIXATIONMAPS"
        if self.phase == "train":
            self.samples = train_samples
        elif self.phase == "val":
            self.samples = val_samples
        elif self.phase == "test":
            self.samples = test_samples
        else:
            self.samples = None

    def file_name_to_path(self, file_name):
        img_path = self.images_dir / (file_name + '.jpg')
        sal_path = self.sals_dir / (file_name + '.jpg')
        #########
        fix_path = self.fixs_dir / (file_name + '_fixPts.jpg')

        return img_path, sal_path, fix_path


class SaliencyDataloder:
    def __init__(self, source, batch_size, phase, out_size, val_rito=0.2, seed=0):
        self.seed = seed
        self.source = source.upper()
        self.dataset = {"SALICON": SALICONDataset,
                        "MIT1003": MIT1003Dataset,
                        "MIT300": MIT300Dataset,}[self.source](source=source, phase=phase, out_size=out_size, val_rito=val_rito)

        self.dataloder = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=16, drop_last=phase == "train", pin_memory=True,
                                         worker_init_fn=self._init_fn)

    def get_dataloder(self):
        return self.dataloder

    def get_dataset(self):
        return self.dataset

    def _init_fn(self, worker_id):
        np.random.seed(int(self.seed) + worker_id)



