import numpy as np
import os
import pyedflib
import torch
from model.models import Epoch


EPOCH_LENGTH = 30


class FeatureConstruction(object):

    @staticmethod
    def construct(sc, rdd, feature_filepath):
        rdd = rdd.filter(lambda x: x.expert_annotation not in ['Sleep stage ?', 'Movement time']) \
            .zipWithIndex()

        data = []
        labels = []
        for epoch, idx in rdd.collect():
            np_epoch = np.zeros((7, EPOCH_LENGTH * 100))
            for channel, signal in epoch.channels.items():
                # Stretch out the signal for the smaller frequencies to match dims
                if epoch.sample_frequencies[channel] == 100:
                    np_epoch[channel + 1] = signal
                elif epoch.sample_frequencies[channel] == 1:
                    np_epoch[channel + 1] = np.repeat(signal, 100, axis=0)
                else:
                    raise Exception('Unexpected sample frequency')
            np_epoch[0] = idx
            data.append(np_epoch)
            labels.append(epoch.expert_annotation)
        data = np.concatenate(data, axis=1)
        np.savez_compressed(feature_filepath, data=data, labels=np.array(labels))


def chunk(signal, sample_frequency):
    num_samples = EPOCH_LENGTH * sample_frequency
    for i in range(0, len(signal), num_samples):
        yield signal[i:i+num_samples]


def extract_features(sc, edf_paths, save=True):
    """
    Read PSGs at edf_paths and construct feature files for them when save flag is True.
    If False, just return feature paths for existing feature files.
    """
    print("Extracting features ...")
    feature_paths = [] 
    if not os.path.exists('./data/features/'):
        os.mkdir('./data/features/')
    for record_idx, (psg_path, hypno_path) in enumerate(edf_paths):
        feature_filepath = './data/features/{}_feature.npz'.format(record_idx)
        if not os.path.exists(feature_filepath):
            if save:
                print("Creating feature file {}".format(feature_filepath))
                epochs = []
                psg_reader = pyedflib.EdfReader(psg_path)
                hyp_reader = pyedflib.EdfReader(hypno_path)

                # Load expert annotations
                # There's expected to be more annotations by epoch than number of epochs for some recordings
                # It's always either an extended W or ? sleep stage at the end of the file ...
                annotations = hyp_reader.readAnnotations()
                annotations_by_epoch = []
                for n in np.arange(hyp_reader.annotations_in_file):
                    annotations_by_epoch.extend([annotations[2][n]] * int(annotations[1][n]/EPOCH_LENGTH))

                # Pivot and construct epochs with their annotations
                for channel in range(psg_reader.signals_in_file - 1):
                    buf = psg_reader.readSignal(channel)
                    sample_frequency = psg_reader.getSampleFrequency(channel)
                    for epoch_id, epoch_signal in enumerate(chunk(buf, sample_frequency)):
                        if epoch_id >= len(epochs):
                            epochs.append(Epoch(epoch_id, annotations_by_epoch[epoch_id]))
                        epochs[epoch_id].add_channel(channel, epoch_signal, sample_frequency)

                rdd = sc.parallelize(epochs)
                FeatureConstruction.construct(sc, rdd, feature_filepath)
            else:
                continue
        feature_paths.append(feature_filepath)
    return feature_paths
        