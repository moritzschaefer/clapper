import glob
import os
import shutil
import logging

import scipy.io.wavfile
import numpy as np

from config import SAMPLE_RATE, SECONDS_BEFORE, SECONDS_AFTER, FFT_SIZE

logging.basicConfig(level=logging.INFO)


class SampleCreator(object):
    """Extract random pieces of the distortion files and save them as files"""

    def __init__(self,
                 raw_distortion_dir,
                 extraced_claps_dir,
                 sample_rate=SAMPLE_RATE,
                 seconds_before=SECONDS_BEFORE,
                 seconds_after=SECONDS_AFTER):
        self._distortion_files = \
            glob.glob(os.path.join(raw_distortion_dir, '*.wav'))
        self._clap_files = glob.glob(os.path.join(extraced_claps_dir, '*.wav'))
        self._sample_rate = sample_rate

        self._distortion_sample_count = \
            (seconds_before + seconds_after) * self._sample_rate

    def extract_distortions(self,
                            num_distortions,
                            target_dir,
                            merge_claps=False):
        """
        Extract distortion and save to directory. optionally merge
        """
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir)
        self._distortion_index = 0
        self.mx = 0

        i = 0
        num_distortions_per_file = \
            num_distortions / len(self._distortion_files)

        for distortion_file in self._distortion_files:
            # read file
            rate, signed_data = scipy.io.wavfile.read(distortion_file)
            if rate != self._sample_rate:
                raise ValueError('Wav file {} has to be sampled in {} Hz'.
                                 format(distortion_file),
                                 self._sample_rate)

            max_start_index = len(signed_data)-self._distortion_sample_count

            start_indices = \
                np.random.randint(0, max_start_index, num_distortions_per_file)

            self._save_distortions(signed_data,
                                   start_indices,
                                   target_dir,
                                   merge_claps)

            i += 1
        print(self.mx)

    def _merge_clap_data(self, data):
        clap_file = np.random.choice(self._clap_files)
        _, clap_data = scipy.io.wavfile.read(clap_file)
        start_index = np.random.randint(0, FFT_SIZE)
        clap_data = np.concatenate((np.array(start_index * [0],
                                             dtype=np.int16),
                                    clap_data))[:len(data)]

        # combined = audioop.add(data, clap_data, SAMPLE_WIDTH)
        # TODO a lot of overflows here: BAD!!
        return data + clap_data

    def _save_distortions(self,
                          data,
                          start_indices,
                          target_dir,
                          merge_claps):

        for index in start_indices:
            sample_data = data[index:index+self._distortion_sample_count]

            if merge_claps:
                sample_data = self._merge_clap_data(sample_data)
                target_filename = \
                    'distortion_clap_{}.wav'.format(self._distortion_index)

                # sample_data[sample_data >= (2**15)] = 2**15-1
                # sample_data[sample_data <= -(2**15)] = -(2**15)+1

            else:
                target_filename = \
                    'distortion_{}.wav'.format(self._distortion_index)

            if max(sample_data) > self.mx:
                self.mx = max(sample_data)

            scipy.io.wavfile.write(
                os.path.join(
                    target_dir,
                    target_filename
                ),
                self._sample_rate,
                sample_data)

            self._distortion_index += 1


if __name__ == "__main__":
    from config import NUM_DISTORTION_SAMPLES, RAW_DISTORTION_DIR, \
        NEGATIVE_SAMPLES_DIR, POSITIVE_SAMPLES_DIR, EXTRACTED_CLAP_DIR
    sc = SampleCreator(RAW_DISTORTION_DIR, EXTRACTED_CLAP_DIR)
    sc.extract_distortions(NUM_DISTORTION_SAMPLES,
                           NEGATIVE_SAMPLES_DIR,
                           False)
    sc.extract_distortions(NUM_DISTORTION_SAMPLES,
                           POSITIVE_SAMPLES_DIR,
                           True)
