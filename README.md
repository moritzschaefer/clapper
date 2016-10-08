# Problem

Voice recognition and voice control is a very active research field [andrew ng at bing, ..., ...]. Different companies have already tried to bring solutions [amazon echo, google now, apple siri, ...]. Though, until now, they didn't reach a broad consumer mass [statistics of voice control usage]. A need for contact-less control of computer system is still very attractive and desired by many end-consumers [find statistic]. A major blocker in the success of voice control system seems to be natural language processing. Algorithms are still not good enough to get the real meaning of our language and as such we can just use pre-defined commands. This can lead to frustration, as end-consumer can't guess which commands work properly and in what way.

# Solution

There is no need for our computers to understand the whole meaning of what we say, as we usually want to do them specific commands. These can be defined beforehand in an easy detectable way: Hand Clapping.


This project contains everything related to my clap detection. Consult the distinct directories for details

# Parameters

Defined in config.py

Samlerate: 44100 Hz
FFT Window size: 512 samples
Clap length: 0.17 seconds


# Strategy

## Test/Training Set

I recorded different distrotion

1. Clap extraction
2. Distortion extraction
  Randomly 10000 pieces of length "clap length" are cut of the distortions
3. Merging of claps and distortion
  Each of these distortion files will be used twice: Once as negative training sample. The second time it is merged with a random clap and used as a positive training sample. The clap is merged with a random offset between 0 and 512 (FFT size).

## Feature extraction

rFFT

TODO: is len(data) == len(np.fft.rfft(data))????

np.log10(np.abs(np.fft.rfft(data))**2)

TODO: experiment whether np.log10 is good or bad

## Maybe condense features (PCA)

##


