import glob
import os
import shutil
import logging

import scipy.io.wavfile
import numpy as np
# import matplotlib.pyplot as plt

from config import SAMPLE_RATE, WINDOW_SIZE, WINDOW_STEP, SECONDS_BEFORE, \
    SECONDS_AFTER

logging.basicConfig(level=logging.INFO)

# for each remaining index in list, extract 0.1 seconds before and 0.2 seconds
# after from original sound file and save as new soundfile in data directory


class ClapExtractor(object):
    """Extract Claps from audio files and save as files"""

    def __init__(self,
                 raw_clap_dir,
                 sample_rate=SAMPLE_RATE,
                 seconds_before=SECONDS_BEFORE,
                 seconds_after=SECONDS_AFTER):
        self._files = glob.glob(os.path.join(raw_clap_dir, '*.wav'))
        self._sample_rate = sample_rate

        self._samples_before = seconds_before * self._sample_rate
        self._samples_after = seconds_after * self._sample_rate

    def extract_claps(self, extracted_claps_dir, threshold_percentile):
        """
        Extract clap and save to directory
        """
        shutil.rmtree(extracted_claps_dir, ignore_errors=True)
        os.makedirs(extracted_claps_dir)
        self._clap_index = 0

        i = 0
        for clap_file in self._files:
            # read file
            rate, signed_data = scipy.io.wavfile.read(clap_file)
            if rate != self._sample_rate:
                raise ValueError('Wav file {} has to be sampled in {} Hz'.
                                 format(clap_file),
                                 self._sample_rate)

            data = np.abs(signed_data)
            peaks = self._get_peaks(data, threshold_percentile)
            # plot these indices

            distilled_peaks = self._distill_peaks(data, peaks)

            logging.info('Found {} peaks in {}. Saving...'.format(
                sum(distilled_peaks), clap_file
            ))
            self._save_claps(signed_data, peaks, extracted_claps_dir)

            i += 1

    def _save_claps(self, data, peaks, extracted_claps_dir):
        for i in np.nonzero(peaks)[0]:
            clap_data = data[i-self._samples_before:i+self._samples_after]

            scipy.io.wavfile.write(
                os.path.join(
                    extracted_claps_dir,
                    'clap_{}.wav'.format(self._clap_index)
                ),
                self._sample_rate,
                clap_data)

            self._clap_index += 1

    def _get_peaks(self, data, threshold_percentile):
        # - use top n percent as threshold.
        # - create a list with all indices of points above threshold
        # - plot the indices
        clap_threshold = np.percentile(data, threshold_percentile)
        return data > clap_threshold

    # FIXME TODO bad practice to use SAMPLE_RATE! use self._sample_rate
    def _distill_peaks(self,
                       data,
                       peaks,
                       window_size=int(WINDOW_SIZE * SAMPLE_RATE),
                       window_step=int(WINDOW_STEP * SAMPLE_RATE)):
        """ Extracts only one point per peak.
            Window over data (0.7), hoplength: 0.4, delete all but the max

        :data: The original signal
        :peaks: The recognized peak
        :returns: Distilled peaks

        """

        distilled_peaks = peaks[:]
        for window_slice in [slice(i, i+window_size) for i in
                             range(0, len(data)-window_size, window_step)]:
            max_index = np.argmax(data[window_slice])
            if distilled_peaks[window_slice][max_index]:
                distilled_peaks[window_slice].fill(False)
                distilled_peaks[window_slice][max_index] = True

        return distilled_peaks


if __name__ == "__main__":
    from config import RAW_CLAP_DIR, EXTRACTED_CLAP_DIR, EXTRACT_THRESHOLD
    ce = ClapExtractor(RAW_CLAP_DIR)
    ce.extract_claps(EXTRACTED_CLAP_DIR, EXTRACT_THRESHOLD)
