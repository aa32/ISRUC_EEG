class Epoch(object):

    def __init__(self, epoch_id, expert_annotation=None):
        self.epoch_id = epoch_id
        self.expert_annotation = expert_annotation
        self.channels = {}
        self.sample_frequencies = {}
    
    def add_channel(self, channel, signals, sample_frequency):
        self.channels[channel] = signals
        self.sample_frequencies[channel] = sample_frequency
    
    def bytesize(self):
        n = 0
        for v in self.channels.values():
            n += v.nbytes
        return n