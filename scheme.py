import numpy as np

from fuel.schemes import SequentialScheme

from picklable_itertools import imap
from picklable_itertools.extras import partition_all


# since the targets are one timestep away from
# the training the data, we need a scheme that overlaps
# on one timestep
class OverlapSequentialScheme(SequentialScheme):
    def __init__(self, *args, **kwargs) :
        self.first_pass = True
        super(OverlapSequentialScheme, self).__init__(*args, **kwargs)


    def get_request_iterator(self):
        if self.first_pass :
            self.batch_size += 1

            size = len(self.indices)
            idlist = np.array(range(size))
            nb_overlaps = size / (self.batch_size-1)

            new_idlist = np.zeros(size+nb_overlaps, dtype=np.int32)
            j = -1
            k = 0
            for i in range(len(new_idlist)):
                if k == self.batch_size and j + self.batch_size-1 > size :
                    break
                j += 1
                if k == self.batch_size :
                    j -= 1
                    k = 0
                new_idlist[i] = idlist[j]
                k += 1
            dellist = (new_idlist == 0)
            dellist[0] = False
            dellist = (dellist == False)

            new_idlist = np.ndarray.tolist(new_idlist[dellist])
            self.indices = new_idlist
            self.first_pass = False

        return imap(list, partition_all(self.batch_size, self.indices))
