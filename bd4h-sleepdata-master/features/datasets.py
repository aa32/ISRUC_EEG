import bisect
import numpy as np
import torch
import time
from features.etl_edf import EPOCH_LENGTH
from operator import itemgetter
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import Sampler


class RecordSampler(Sampler):
    '''
    Randomly sample in one record before moving onto the next to avoid excessive loading.
    '''

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        idxs = np.concatenate([torch.randperm(y - x) + x for x, y in self.dataset.epoch_ranges])
        return iter(idxs)

    def __len__(self):
        return len(self.dataset)


class WeightedRecordSampler(Sampler):
    '''
    Sample epochs in each record by weight which is calculated by number of occurrences per class.
    '''

    def __init__(self, dataset, replacement=True):
        self.dataset = dataset
        self.weights = []
        self.replacement = replacement
        for stages in dataset.labels:
            class_occur = np.array([len(np.where(stages == t)[0]) for t in np.arange(max(stages) + 1)])
            class_occur[class_occur == 0] = -1
            class_weights = 1. / class_occur
            samples_weights = np.array([class_weights[stage] for stage in stages])
            samples_weights = torch.from_numpy(samples_weights).double()
            self.weights.append(samples_weights)

    def __iter__(self):
        idxs_per_range = []
        for i, (start, end) in enumerate(self.dataset.epoch_ranges):
            idxs = torch.multinomial(self.weights[i], end - start, self.replacement).tolist()
            idxs = [idx + start for idx in idxs]
            idxs_per_range.append(idxs)
        return iter(np.concatenate(idxs_per_range))

    def __len__(self):
        return len(self.dataset)


class EpochRangeKeyList(object):
    '''
    Wrapper to bisect epoch ranges in feature files to find which file contains particular epoch.
    '''

    def __init__(self, l, key):
        self.l = l
        self.key = key

    def __getitem__(self, index):
        return self.key(self.l[index])

    def __len__(self):
        return len(self.l)


class EpochDataset(Dataset):
    '''
    Epoch is (6 channel x 3000 sample) tensor. Randomly sampled from each feature file.
    This might not be so good? Tsinalis paper uses 1D raw signal input (1x15000) but not sure how
    it's constructed. Maybe just appended each channel to each other? Biswal paper creates
    spectograms which is an interesting approach.
    '''

    def __init__(self, feature_paths, class_map):
        self.files = feature_paths
        self.cache = {}
        self.epoch_ranges = []
        self.labels = []

        onset = 0
        for feature_file in self.files:
            f = np.load(feature_file)
            data = f['data']
            self.labels.append(list(map(class_map.get, f['labels'])))
            num_epochs = int(data[0][-1] + 1)
            self.epoch_ranges.append((onset, onset + num_epochs))
            onset += num_epochs
        self.key_list = EpochRangeKeyList(self.epoch_ranges, key=lambda x: x[0])

    def __getitem__(self, epoch_idx):
        # find feature file for this particular epoch
        file_idx = bisect.bisect_right(self.key_list, epoch_idx) - 1
        if file_idx not in self.cache:
            # cache one file at a time
            f = np.load(self.files[file_idx])
            self.cache = {file_idx : f['data']}
        data = self.cache[file_idx]
        labels = self.labels[file_idx]

        # subtract onset to get epoch idx within this file
        rel_idx = epoch_idx - self.epoch_ranges[file_idx][0]
        data = data[1:, rel_idx*EPOCH_LENGTH*100:(rel_idx+1)*EPOCH_LENGTH*100]

        # collapse into 1d signal like in tsinalis paper
        data = np.expand_dims(np.concatenate(data[:]), axis=0)

        target = labels[rel_idx]
        return data.astype(np.float32), target

    def __len__(self):
        return self.epoch_ranges[-1][1]


class RNNDataset(EpochDataset):
    
    def __init__(self, feature_paths, class_map):
        super(RNNDataset, self).__init__(feature_paths, class_map)

    def __getitem__(self, epoch_idx):
        # find feature file for this particular epoch
        file_idx = bisect.bisect_right(self.key_list, epoch_idx) - 1
        if file_idx not in self.cache:
            # cache one file at a time
            f = np.load(self.files[file_idx])
            self.cache = {file_idx : f['data']}
        data = self.cache[file_idx]
        labels = self.labels[file_idx]

        # subtract onset to get epoch idx within this file
        rel_idx = epoch_idx - self.epoch_ranges[file_idx][0]
        data = data[1:, rel_idx*EPOCH_LENGTH*100:(rel_idx+1)*EPOCH_LENGTH*100]

        data = np.average(data, axis=0)
        data = np.reshape(data, (1,3000))
        data = np.expand_dims(np.concatenate(data[:]), axis=1)

        target = labels[rel_idx]
        return data.astype(np.float32), target
