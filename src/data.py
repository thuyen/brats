import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
import os
import math
import multiprocessing as mp
import threading
import torch
from torch.utils.data import Dataset

import numpy as np
from utils import get_receptive_field, get_sub_patch_shape, \
        get_offset, sample_coords, get_all_coords, nib_load

PATCH_SHAPE = (25, 25, 25)
KERNELS = ((3, 3, 3), )*8
SCALE_FACTOR = (3, 3, 3)
SHAPE = [240, 240, 155]

np.random.seed(2017)

class ImageList(Dataset):
    def __init__(self,
            list_file,
            patch_shape=PATCH_SHAPE,
            kernels=KERNELS,
            scale_factor=SCALE_FACTOR,
            root='',
            split='valid',
            sample_size=500):

        with open(list_file) as f:
            names = f.read().splitlines()
            names = [os.path.join(root, name) for name in names]

        self.root = root
        self.names = names
        self.split = split
        self.sample_size = sample_size
        self.receptive_field = get_receptive_field(kernels)
        self.patch_shape = np.array(patch_shape)
        self.scale_factor = np.array(scale_factor)
        self.sub_patch_shape = get_sub_patch_shape(self.patch_shape,
                self.receptive_field, self.scale_factor)
        self.sub_off = get_offset(self.scale_factor, self.receptive_field)
        self.modalities = ('Flair', 'T1c', 'T1', 'T2')
        self.C = len(self.modalities)

    def coord_to_slice(self, coord):
        return coord[:, 0], coord[:, 1] + 1

    def coord_to_sub_slice(self, coord):
        lo = coord[:, 0] + self.sub_off
        num = self.patch_shape - self.receptive_field+ 1
        hi = lo + self.scale_factor*self.receptive_field + \
                np.ceil((num*1.0)/self.scale_factor - 1) * self.scale_factor
        hi = hi.astype('int')

        m = lo < 0
        pl = -lo * m
        lo[lo < 0] = 0

        m = hi > SHAPE
        ph = (hi - SHAPE) * m
        hi += pl

        pad = zip(pl, ph)
        return lo, hi, pad

    def crop(self, coords, label, path):
        N = coords.shape[0]
        samples = np.zeros((N, self.C) + tuple(self.patch_shape), dtype='float32')
        sub_samples = np.zeros((N, self.C) + tuple(self.sub_patch_shape), dtype='float32')
        labels = np.zeros((N,) + (9, 9, 9), dtype='int')

        size = (self.sub_patch_shape - 1)/2
        gl = (self.patch_shape - size)/2
        gh = self.patch_shape - gl

        kx, ky, kz = self.scale_factor

        images = np.array([
            nib_load(os.path.join(path, modal + '_subtrMeanDivStd.nii.gz')) \
            for modal in self.modalities])

        for n, coord in enumerate(coords):
            ss, ee = self.coord_to_slice(coord)
            lo, hi, pad = self.coord_to_sub_slice(coord)

            cropped_label = label[ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2]]
            labels[n] = cropped_label[gl[0]:gh[0], gl[1]:gh[1], gl[2]:gh[2]]

            samples[n] = images[:, ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2]]

            pimages = np.pad(images, [(0, 0)] + pad, mode='constant')
            sub_samples[n] = \
                    pimages[:, lo[0]:hi[0]:kx, lo[1]:hi[1]:ky, lo[2]:hi[2]:kz]

        return samples, sub_samples, labels


    def __call__(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        path = self.names[index]

        mask_file = os.path.join(path, 'brainmask.nii.gz')
        mask = nib_load(mask_file)
        label_file = os.path.join(path, 'OTMultiClass.nii.gz')
        label = nib_load(label_file)

        n = self.sample_size
        if self.split == 'train':
            fg = (label > 0).astype('int32')
            bg = ((mask > 0) * (fg == 0)).astype('int32')
            coords = np.concatenate(
                    [sample_coords(n/2, self.patch_shape, weight) for weight in (fg, bg)])
        elif self.split == 'valid':
            coords = sample_coords(n, self.patch_shape, mask)
        else: # test
            coords = get_all_coords((9, 9, 9), self.patch_shape, SHAPE, 15)
        samples, sub_samples, labels = self.crop(coords, label, path)

        return samples, sub_samples, labels, coords


    def __len__(self):
        return len(self.names)


class MemTuple(Dataset):
    def __init__(self, data):
        for k, datum in enumerate(data):
            size = datum.size()
            new_size = (size[0]*size[1], ) + size[2:]
            datum = datum.view(*new_size)
            data[k] = datum
        self.data = data

    def __getitem__(self, index):
        return [d[index] for d in self.data]

    def __len__(self):
        return len(self.data)

class PEDataLoader(object):
    """
    A multiprocess-dataloader that parallels over elements as suppose to
    over batches (the torch built-in one)
    Input dataset must be callable with index argument: dataset(index)
    https://github.com/thuyen/nnet/blob/master/pedataloader.py
    """

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=None, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.collate_fn = torch.utils.data.dataloader.default_collate
        self.pin_memory_fn = \
                torch.utils.data.dataloader.pin_memory_batch if pin_memory else \
                lambda x: x

        self.num_samples = len(dataset)
        self.num_batches = int(math.ceil(self.num_samples / float(self.batch_size)))

        self.pool = mp.Pool(num_workers)
        self.buffer = queue.Queue(maxsize=1)
        self.start()

    def generate_batches(self):
        self.indices = \
                torch.randperm(self.num_samples).long() if self.shuffle else \
                torch.LongTensor(range(self.num_samples))

        for b in range(self.num_batches):
            start_index = b*self.batch_size
            end_index = (b+1)*self.batch_size if b < self.num_batches - 1 \
                    else self.num_samples
            indices = self.indices[start_index:end_index]
            batch = self.pool.map(self.dataset, indices)
            batch = self.collate_fn(batch)
            batch = self.pin_memory_fn(batch)
            yield batch

    def start(self):
        def _thread():
            for b in self.generate_batches():
                self.buffer.put(b, block=True)
            self.buffer.put(None)

        thread = threading.Thread(target=_thread)
        thread.daemon = True
        thread.start()

    def __next__(self):
        batch = self.buffer.get()
        if batch is None:
            self.start()
            raise StopIteration
        return batch

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

