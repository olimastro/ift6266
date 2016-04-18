import numpy as np

from fuel.schemes import BatchScheme

from picklable_itertools import imap
from picklable_itertools.extras import partition_all


class ShuffledBatchChunkScheme(BatchScheme):
    def get_request_iterator(self):
        chunks = len(self.indices) / self.batch_size
        assert len(self.indices)%chunks == 0

        data = np.array(self.indices)
        data = data.reshape(chunks, self.batch_size)
        np.random.shuffle(data)
        data = data.flatten()

        self.indices = np.ndarray.tolist(data)

        return imap(list, partition_all(self.batch_size, self.indices))
