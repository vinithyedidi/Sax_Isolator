from sklearn.decomposition import non_negative_factorization as nmf
from scipy.fft import fft, ifft, fftfreq
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np


'''
Method to read audio data from .wav file and convert it into a normalized, monophonic array of sample data, y.
'''
def read_audio(filename):
    sr, rawdata = read(filename)

    # converts to mono if not already mono
    y = np.zeros((rawdata.shape[0]))
    numchannels = rawdata.shape[1]
    if numchannels == 1:
        y = rawdata
    else:
        for n in range(numchannels):
            y = y + rawdata[:, n]
        y = y/numchannels

    # normalize amplitude to 1
    y = y/y.max()

    dur = y.size // sr
    return y, dur, sr


'''
Method to decompose sample array into its Fourier Spectrogram matrix, and then decompose that using NMF to 
two matrices, D and H. Cols of D are bases for the sources, and rows of H are weights for the expression 
of the bases. This also retains phase information for lossless reconstruction of the original audio.
'''
def decompose(y, k):
    # Apply Short-Time Fourier Transform on y to get Fourier Spectrogram Y
    WIN_LEN = 256
    HOP_LEN = WIN_LEN // 2
    w = np.hanning(WIN_LEN) # experientially, the Hanning window seems to yield the best results
    Y = np.zeros((((y.size-WIN_LEN) // HOP_LEN) + 1, WIN_LEN), dtype='complex128')
    for i in range(0, len(y)-WIN_LEN, HOP_LEN):
        Y[i//HOP_LEN, :] = fft(w * y[i:i+WIN_LEN])
    # NMF only allows positive, real values, so we store phase info separately and decompose the absolute value of Y.
    phase = np.angle(Y)

    # 5000 iterations provides a fairly accurate value without taking too long.
    D, H = nmf(abs(Y), n_components=k, max_iter=5000)[:2]
    return D, H, phase


'''
Method to recompose the NMF matrices back into the original audio. The user selects the number of components, k, they
wish to recompose with, allowing for a sparser or more dense matrix (more sparse -> more accurate but more computation).
'''
def recompose(D, H, phase, mask, dur, sr):
    WIN_LEN = 256
    HOP_LEN = WIN_LEN // 2

    D = D[:, mask]
    H = H[mask, :]
    # Good ol' Euler's formula: e^ix = cos(x) + isin(x)
    Y = np.dot(D, H) * np.exp(1j*phase)

    # Inverse STFT
    y = np.zeros(int(dur * sr))
    for n, i in enumerate(range(0, len(y) - WIN_LEN, HOP_LEN)):
        # Imaginary component is arbitrarily close to zero anyway so minimal loss here.
        y[i:i + WIN_LEN] += np.real(ifft(Y[n]))
    y = y/y.max()

    return y


'''
Method (optional) to check each base of D with each weight in H, plot it, and listen to it. This is useful for
manually examining the bases to determine an appropriate mask to isolate the saxophone. Can be tedious though.
'''
def check_bases(D, H, phase, dur, sr):
    for n in range(D.shape[1]):
        print("Base #" + str(n))
        y = recompose(D, H, phase, [n], dur, sr)
        y = y/y.max()

        t = np.linspace(0, dur, y.size)
        sd.play(y, sr)
        plt.plot(t, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude [-1,1]")
        plt.title("Base #" + str(n) + " of " + str(D.shape[1]-1))
        plt.show()
        sd.stop()


'''
Method to consolidate all previous methods and separate the sax from the rhythm. The mask (for now) is applyed manually,
derived from listening to each base with check_bases(). To brute-force the problem and provide greater accuracy, k is 
set to 50, which makes checking bases tedious. I hope to solve this by automating it with machine learning.
'''
def isolate_sax(filename):
    y, dur, sr = read_audio(filename)
    y = y/y.max()

    k = 50  # 50 components provides adequate sparsity without taking forever to compute - a good compromise.
    D, H, phase = decompose(y, k)

    # The mask for saxophone is listed here, and the inverse mask (rhythm section/everything else) is the compliment.
    # 3, 14, 18, 25, 35, 41, 42, 45, 47, 48, 49 are ambiguous to my perception.
    mask = [2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 28, 29, 30, 32, 34, 37, 38, 39, 41, 45, 46]
    invmask = []
    for x in range(k):
        if x not in mask:
            invmask.append(x)

    # NOTE: uncomment and run check_bases() to examine each base of the decomposition.
    #check_bases(D, H, phase, dur, sr)
    sax = recompose(D, H, phase, mask, dur, sr)
    rhythm = recompose(D, H, phase, invmask, dur, sr)

    return y, sax, rhythm, dur, sr


'''
We test this procedure using one of my favorite saxophone lines, Lady Bird played by Dexter Gordon. The fundamental
issue of source separation with jazz is that the voices blend together extremely well, so the bases of D seem to blend
different voices together, misconstruing them as a single source. You can hear this in the way the piano contaminates
the saxophone source bases. I believe that this can be fixed with sparser decomposition, and machine learning methods
to construct a mask with more elements. However, running this shows a fairly successful proof of concept without using
any machine learning.
'''
original, sax, rhythm, dur, sr = isolate_sax('audio/Lady Bird.wav')
t = np.linspace(0, dur, original.size)

sd.play(original, sr)
plt.plot(t, original)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude [-1,1]")
plt.title("Original Audio")
plt.show()
sd.stop()

t = np.linspace(0, dur, sax.size)
sd.play(sax, sr)
plt.plot(t, sax)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude [-1,1]")
plt.title("Isolated Saxophone")
plt.show()
sd.stop()

t = np.linspace(0, dur, rhythm.size)
sd.play(rhythm, sr)
plt.plot(t, rhythm)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude [-1,1]")
plt.title("Rhythm Section + Noise")
plt.show()
sd.stop()
