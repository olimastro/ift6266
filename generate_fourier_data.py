import numpy as np
import h5py
from scipy.io.wavfile import read
from scipy import signal

WINDOW_SIZE = 512
OVERLAP = 512/2

data = read("/data/lisatmp3/mastropo/XqaJ2Ol5cC4.wav")

samplerate = data[0]
data = data[1]

data = data[samplerate*60*1:len(data)-samplerate*60*1]
window = signal.cosine(WINDOW_SIZE)

f_data = signal.spectrogram(data, window=window, nperseg=WINDOW_SIZE, noverlap=OVERLAP, mode='complex')
f_data = np.swapaxes(f_data[2], 0, 1)
real = np.real(f_data)
rmax = np.max(real) if np.absolute(np.min(real)) < np.max(real) else np.absolute(np.min(real))
real /= rmax
cplx = np.imag(f_data)
cmax = np.max(cplx) if np.absolute(np.min(cplx)) < np.max(cplx) else np.absolute(np.min(cplx))
cplx /= cmax

data = np.append(real, cplx, axis=1)

f = h5py.File('/Tmp/mastropo/fouried_song.hdf5', mode='w')
frequency_sequence = f.create_dataset('frequency_sequence', (data.shape[0], data.shape[1]), dtype='float32')
frequency_sequence[...] = data
frequency_sequence.dims[0].label = 'time'
frequency_sequence.dims[1].label = 'feature'

from fuel.datasets.hdf5 import H5PYDataset
split_dict = {
    'train' : {'frequency_sequence' : (0, data.shape[0])}
}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()
