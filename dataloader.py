from base import *

class DataLoader(object):
    def __init__(self, data, labels, batch_size, shuffle=True):
        self.data = data
        self.labels = labels
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.idxs = np.array([i for i in range(self.data.shape[0])])

        if self.shuffle:
            np.random.shuffle(self.idxs)

    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = idx * self.batch_size + self.batch_size
        batch_idxs = self.idxs[batch_start:batch_end]
        data = self.data[batch_idxs]
        label = self.labels[batch_idxs]
        return {'data':data, 'label':label}