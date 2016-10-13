import time
import logging
import alsaaudio
import struct

import numpy as np

from feature_extraction import transform_to_freq


class Recorder:
    def __init__(self, num_windows, fft_size):
        self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)

        # Set attributes: Mono, 4410- Hz, 16 bit little endian samples
        self.inp.setchannels(1)
        self.inp.setrate(44100)
        self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.inp.setperiodsize(fft_size)

        self._num_windows = num_windows
        self._fft_size = fft_size

        self._windows = np.zeros((num_windows, (fft_size / 2) + 1),
                                 dtype=np.float64)
        self._index = 0

    def read_input(self, cb):
        while True:
            # Read data from device
            l, data = self.inp.read()

            if l:
                # convert to array
                ints = struct.unpack('h'*self._fft_size, data)
                transformed = transform_to_freq(ints)
                if np.isinf(transformed).any():
                    logging.warn('inf detected. Skip window')
                    continue

                self._windows[self._index, :] = transformed
                self._index = (self._index + 1) % self._num_windows
                indices = list(range(self._index, self._num_windows)) + \
                    list(range(0, self._index))
                cb(self._windows[indices, :])
            else:
                logging.warn('Didn\'t find data. strange')
                time.sleep(.001)
