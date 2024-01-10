# Sax Isolator


## **The Problem: AX = Y**
Given a matrix of mixed audio Y (such as Lady Bird by Dexter Gordon), we need to solve for the unmixed samples matrix X. However, we don't know A, the mixing algorithm matrix, making this problem only solvable by estimation rather than excatly. More specifically, we have the entire ensemble mixed into the audio, Lady Bird.wav, and our task is to isolate the saxophone voice out from the rest of the rhythm section and any other residual noise. This has significant applications, such as cutting out the lead voice in a jazz track to practice soloing and improvisation on the original backing track without having to get an external, imitation backing track of a jazz piece. This has the potential to be a great practice tool for saxophone players, such as myself.



## **The Solution: NMF**
To solve the problem and isolate the sax, I employed Non-Negative Matrix Factorization (NMF), a technique that decomposes Y into DH, where the columns of D are bases representing different sources of audio, and the rows of H are the weights at which bases in D are expressed. This is lossy, but with sufficiently many components (I chose k = 50) and sufficiently many iterations of the algorithm to compute D and H (I use 5000 iterations), we can make DH converge to Y using the Küllbach-Leibler Divergence as a metric of difference.

Then, I manually selected bases (according to how they sound and appear in waveform) that correspond to the sax or the rhythm section. This was a tedious process, and only possible at low numbers of components, but in the future I hope to automate this using a machine learning algorithm. The result is two separate audio streams, one with roughly the saxophone (with some contamination from higher frequencies in the rhythm section) and the other with roughly the rhythm section (with some contamination from Dexter's beautiful lower frequencies). With sparser NMF matrices and a machine learning algorithm, I hope to reduce this contamination further.

It's also important to note that NMF requires non-negative values in the Fourier spectrogram Y, but to convert audio into the spectrogram, it involves a Short-Time Fourier Transform resulting in complex values. I store this in a separate phase array, which is then composed elementwise to Y using Euler's Formula (e^ix = cos(x) + isin(x)) to reconstruct the audio with minimal loss. This way, NMF can take place without negative values, but phase information isn't lost in the process and audio fidelity is preserved.



## **Using This Function**
You can download IsolateSax.py and Lady Bird.wav to run this program in Python 3.12. The modules I used are sklearn, scipy, matplotlib, sounddevice, and numpy, so you will need to have those installed to use the function. Or, I have uploaded sample graphs of the waveforms and their respective audios so you can just listen to those.



## **Conclusions**
Overall, I would consider this a success in the sense that selecting bases are manual and with limited computational power and manual capacity, this roughly isolates the saxophone from the rhythm section. However, in the future, I hope to implement a machine learning algorithm to select bases of D, allowing for a greater number of components and greater accuracy in choosing them. If that is successfully implemented, then this could become a viable practice tool for accessing original backing tracks over imitations on Youtube.



## **Sources**
- http://www.mit.edu/~gari/teaching/6.555/LECTURE_NOTES/ch15_bss.pdf
- https://www.cs.jhu.edu/~ayuille/courses/Stat161-261-Spring14/HyvO00-icatut.pdf
- Wang and Plumbley. "Musical Audio Stream Separation by Non-Negative Matrix Factorization," (2005).
- Schmidt, Larsen and Hsiao "Wind Noise Reduction using Non-Negative Sparse Coding," (2007).


##
Vinith Yedidi, Jan 10 2024
