import numpy as np
import h5py
from scipy.io.wavfile import read

data = read("/data/lisatmp3/mastropo/XqaJ2Ol5cC4.wav")

samplerate = data[0]
data = data[1]

data = data[samplerate*60*1:len(data)-samplerate*60*1]
data = data.astype(np.float32)

dmax = np.max(np.absolute(data))
data /= dmax

data = data.reshape(data.shape[0], 1)

f = h5py.File('/Tmp/mastropo/song.hdf5', mode='w')
frequency_sequence = f.create_dataset('time_sequence', (data.shape[0], data.shape[1]), dtype='float32')
frequency_sequence[...] = data
frequency_sequence.dims[0].label = 'time'
frequency_sequence.dims[1].label = 'feature'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train' : {'time_sequence' : (0, data.shape[0])}
}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()
