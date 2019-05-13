'''
Feature extraction main
'''

#--- Import
import os
import sys
import platform
sys.path.append(".")
sys.path.append("../data")
sys.path.append("../EMO-DB/wav")

import matplotlib
if platform.system() == 'Darwin':
	import appnope
	appnope.nope()
	matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy
from scipy import stats, signal
import pandas as pd
import math
from numpy import NaN, Inf, arange, isscalar, array
from scipy import linalg as la
from scipy.fftpack import rfft, fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve, freqz, lfilter, hamming
from scipy.stats import kurtosis
from scikits.talkbox import lpc

from pyAudioAnalysis import audioFeatureExtraction as af
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import peakutils

from pyAudioAnalysis import audioBasicIO


"""
monkey-patch crs_corr in pYAAPT.py to prevent from division by 0
"""

eps = np.finfo(float).eps

def crs_corr(data, lag_min, lag_max):
	# eps1 = 0.0
	eps1 = np.finfo(float).eps  # modified
	data_len = len(data)
	N = data_len - lag_max

	error_str = 'ERROR: Negative index in the cross correlation calculation of '
	error_str += 'the pYAAPT time domain analysis. Please try to increase the '
	error_str += 'value of the "tda_frame_length" parameter.'
	assert N > 0, error_str

	phi = np.zeros((data_len))
	data -= np.mean(data)
	x_j = data[0:N]
	x_jr = data[lag_min:lag_max + N]
	p = np.dot(x_j, x_j)

	x_jr_matrix = pYAAPT.stride_matrix(x_jr, lag_max - lag_min, N, 1)  # modified

	formula_nume = np.dot(x_jr_matrix, x_j)
	formula_denom = np.sum(x_jr_matrix * x_jr_matrix, axis=1) * p + eps1

	phi[lag_min:lag_max] = formula_nume / np.sqrt(formula_denom)

	return phi


pYAAPT.crs_corr = crs_corr


def stFeatureExtraction_modified(signal, fs, winLen, winStep):
	"""
	Modification of stFeatureExtraction in pyAudioAnalysis (#0 - 20)
	& Addition of more features (#21 -)

	ARGUMENTS
	    signal:       the input signal samples
	    fs:           the sampling freq (in Hz)
	    winLen:          the short-term window size (in samples)
	    winStep:         the short-term window step (in samples)
	RETURNS
	    stFeatures:   a numpy array (numOfFeatures x numOfShortTermWindows)

		0: zcr
		1: log energy
		2: frame energy entropy
		3: spectral centroid
		4: spectral spread
		5: spectral entropy
		6: spectral flux
		7: spectral rolloff
		8-20: MFCC
		21: probV
		22: pitch
		23-25: formant frequency
		26-28: formant BW
		29-31: formant Gain
		32-36: harmonics peak power (at F0 ~ F4)
		37-39: spectral crest factor in formant
		40-43: spectral crest factor in frequency bands [100, 500], [500, 1000], [1000, 2000], [2000, 4000]

	"""

	lpcOrder = 8

	numOfTimeSpectralFeatures = 8
	nceps = 13
	nFormant = 3
	nFormantFeatures = nFormant * 4
	nfbandCrest = 4
	nHarmonic = 5
	totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + 1 + nFormantFeatures + nfbandCrest + 1 + nHarmonic

	winLen = int(winLen)
	winStep = int(winStep)

	# Signal normalization
	signal = signal.astype(float)
	# signal = signal / (2.0 ** 15)
	# DC = signal.mean(); MAX = (np.abs(signal)).max(); signal = (signal - DC) / MAX

	N = len(signal)  # total number of samples
	curPos = 0
	countFrames = 0
	sizeFFT = 2048
	nFFT = sizeFFT / 2
	ff = float(fs) / 2.0 * np.array(range(0, nFFT)).astype(float) / float(nFFT)

	[fbank, freqs] = af.mfccInitFilterBanks(fs,
	                                        nFFT)  # compute the triangular filter banks used in the mfcc calculation
	stFeatures = np.array([], dtype=np.float64)
	hammW = hamming(winLen)

	# pitch
	signalo = basic.SignalObj(signal, float(fs))
	# pitch = pYAAPT.yaapt(signalo, **{'f0_min' : 60.0, 'f0_max' : 600.0, 'frame_length' : 30.0, 'frame_space': 5.0, 'nlfer_thresh1': 0.95})
	pitch = pYAAPT.yaapt(signalo, **{'f0_min': 80.0, 'f0_max': 400.0, 'frame_length': 30.0, 'frame_space': 10.0,
	                                 'nlfer_thresh1': 0.75})
	# vuv = pitch.vuv[::2]    # 0 or 1 values
	# pitch = pitch.samp_values[::2]
	pitch = pitch.samp_values

	nFrame = len(pitch)

	# while (curPos + winLen - 1 < N):                        # for each short-term window until the end of signal
	for countFrames in range(0, nFrame):
		#print countFrames
		curPos = countFrames * winStep
		if countFrames == 14:
			print countFrames

		x = signal[curPos:curPos + winLen]  # get current window
		x = x * hammW
		X = abs(fft(x, sizeFFT))  # get fft magnitude
		X = X[0:nFFT]  # normalize fft
		X = X / len(X)
		XdB = 20. * np.log10(X + eps)
		if countFrames == 0:
			Xprev = X.copy()  # keep previous fft mag (used in spectral flux)
		curFV = np.zeros(totalNumOfFeatures)
		curFV[0] = af.stZCR(x)  # zero crossing rate
		curFV[1] = 10. * np.log10(af.stEnergy(x) + eps)  # short-term energy
		curFV[2] = af.stEnergyEntropy(x)  # short-term entropy of energy
		[curFV[3], curFV[4]] = stSpectralCentroidAndSpread_modified(X, fs)  # spectral centroid and spread
		curFV[5] = af.stSpectralEntropy(X)  # spectral entropy
		curFV[6] = af.stSpectralFlux(X, Xprev)  # spectral flux
		curFV[7] = af.stSpectralRollOff(X, 0.90, fs)  # spectral rolloff
		curFV[8:8 + nceps] = af.stMFCC(X, fbank, nceps).copy()  # MFCCs

		if pitch[countFrames] != 0:

			# pv
			# https://books.google.com/books?id=AFBECwAAQBAJ&pg=PA62&lpg=PA62&dq=voicing+probability+autocorrelation&source=bl&ots=Oh4WfjJplw&sig=v9qRLiWAjFvnqZBzmFThf_ADy4w&hl=en&sa=X&ved=0ahUKEwiIisfe5uHSAhVX62MKHRr_CpEQ6AEIXTAJ#v=onepage&q=voicing%20probability%20autocorrelation&f=false
			idx = numOfTimeSpectralFeatures + nceps  # 21
			acf = np.correlate(x, x, mode='same')
			a0 = x.shape[0] / 2  # zero-lag position
			idxPeak = a0 + int(round(fs / pitch[countFrames]))
			curFV[idx] = acf[idxPeak] / acf[a0]
			# idxPeak = peakutils.indexes(acf[a0+5:], 0.05) + a0 + 5
			# if len(idxPeak):
			#	curFV[idx] = max(acf[idxPeak]) / acf[a0]
			# else:
			#	curFV[idx] = 0.

			# pitch

			idx += 1  # 22
			curFV[idx] = pitch[countFrames]

			# formants
			formantF, formantBw, formantG, formantCrest, fbandCrest = lpc2formant(x, fs, lpcOrder=lpcOrder)
			nFormantFound = formantF.shape[0]  # number of found formants

			idx += 1  # 23
			curFV[idx:idx + nFormant] = np.copy(formantF[0:nFormant])

			idx += nFormant  # 26
			curFV[idx:idx + nFormant] = np.copy(formantBw[0:nFormant])

			idx += nFormant  # 29
			curFV[idx:idx + nFormant] = np.copy(formantG[0:nFormant])

			# harmonic peak power at F0 ~ F4
			idx += nFormant  # 32
			curFV[idx:idx + nHarmonic] = np.copy(get_harmonicPower(pitch[countFrames], ff, X, nHarmonic))

			# crest factor in formants
			idx += nHarmonic  # 37
			curFV[idx:idx + nFormant] = np.copy(formantCrest[0:nFormant])

			# crest factor in 4 frequency bands
			idx += nFormant  # 40
			curFV[idx:idx + nfbandCrest] = np.copy(fbandCrest)

		if countFrames == 0:
			stFeatures = curFV  # initialize feature matrix (if first frame)
		else:
			stFeatures = np.vstack((stFeatures, curFV))  # update feature matrix

		Xprev = X.copy()

	return np.array(stFeatures)


def get_harmonicPower(f0, ff, X, nharmonic):
	''' return spectral power of nharmonic harmonics in X.
	    ff is an numpy array of frequency for X.
    '''

	freqSpacing = ff[2] - ff[1]
	nSearch = int((2. * f0 / 3.) / freqSpacing)
	hPower = np.zeros(nharmonic)

	for n in range(0, nharmonic):
		freq = f0 * (n + 1)
		idx = np.argmin(np.abs(ff - freq))
		data = X[idx - nSearch:idx + nSearch]
		# plt.clf(); plt.plot(data,'o-'); plt.grid()
		idxPeak = peakutils.indexes(data, 0.05)
		if idxPeak.size != 0:
			hPower[n] = np.max(data[idxPeak])
		else:
			# print("No harmonic peak was found. Fix the problem.")
			hPower[n] = np.max(data)  # simply take maximum value in this case

		'''
        if ff[idx] < freq:
            n1 = idx
            n2 = idx + 1
        else:
            n1 = idx - 1
            n2 = idx
        hPowerdB[n] = XdB[n1] + (XdB[n2] - XdB[n1]) / (ff[n2] - ff[n1]) * (freq - ff[n1])
        '''
	return hPower


def lpc2formant(x, fs, lpcOrder=12):
	# http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html?requestedDomain=www.mathworks.com

	maxnFormant = 3
	minFormantFreq = 90. / float(fs)
	maxFormantFreq = 5000. / float(fs)
	maxFormantBw = 700. / float(fs)

	formantF = np.zeros(maxnFormant)
	formantBw = np.zeros(maxnFormant)
	formantG = np.zeros(maxnFormant)

	# x1 = lfilter([1],[1., 0.63], x)	# this is wrong
	# x1 = lfilter([1., -0.95], 1., x)  # pre-emphasis filter
	x1 = lfilter([1., -0.97], 1., x)  # pre-emphasis filter
	'''
	plt.clf(); plt.plot(np.log10(abs(fft(x))[0:len(x)/2]))
	plt.plot(np.log10(abs(fft(x1))[0:len(x)/2]))
	ww, hh = freqz(1,[1., 0.63])
	ww, hh = freqz([1., -0.95], 1.)
	plt.clf(); plt.plot(float(fs)*ww/(2.*math.pi), 20 * np.log10(abs(hh)), 'b')

	'''
	A, e, k = lpc(x1, lpcOrder)
	'''
	# FFT & LPC spectrum
	ww, hh = freqz(1,A)
	plt.clf(); plt.subplot(2,1,1) 
	plt.plot(float(fs)*ww/(2.*math.pi), 20 * np.log10(abs(hh)), 'b')
	frq=float(fs)/2.0*np.array(range(0,len(x)/2)).astype(float)/(len(x)/2.)
	plt.plot(frq, 20.*np.log10(abs(fft(x1))[0:len(x)/2]))

	ext=lfilter(A,1.0,x1)	# excitation signal
	Ext=20*np.log10(abs(fft(ext))[0:len(x1)/2]); 
	plt.subplot(2,1,2); plt.plot(frq,Ext)
	'''

	# Formant bandwidth
	rts = np.roots(A)
	rts = [r for r in rts if np.imag(r) > 0]  # original: rts = [r for r in rts if np.imag(r) >= 0]
	angz = np.arctan2(np.imag(rts), np.real(rts))
	indices = sorted(range(len(angz)), key=lambda k: angz[k])
	frqs = np.array(sorted(angz)) / (2. * math.pi)
	bw = -2.0 * (1. / (2. * math.pi)) * np.log(
		abs(np.array(rts)[indices]))  # 2.0 makes the result same as in matlab voicebox

	# Formant gain
	ap = A.reshape(1, 1, len(A))
	pw = -2.0 * math.pi * 1j * np.array(range(0, lpcOrder + 1))
	pw = pw.reshape(1, 1, len(pw))

	l = [0 for i in range(0, len(frqs))]
	p1 = [0 for i in range(0, lpcOrder + 1)]
	q1 = ap[:, l, :]
	q2 = pw[0, l, :]
	q3 = frqs.reshape(len(frqs), 1)
	q4 = np.exp(q2 * q3[:, p1])
	gain = np.divide(1.0, np.abs(np.sum(q1 * q4, axis=2)))
	gain = gain.reshape(gain.shape[1])

	# Select valid formants only
	cnt = 0
	for m in range(0, len(frqs)):
		if minFormantFreq < frqs[m] and frqs[m] < maxFormantFreq and bw[m] < maxFormantBw and cnt < maxnFormant:
			formantF[cnt] = frqs[m] * float(fs)
			formantBw[cnt] = bw[m] * float(fs)
			formantG[cnt] = gain[m]
			cnt += 1

	nFormant = cnt
	'''
	formantF = formantF[0:cnt]
	formantBw = formantBw[0:cnt]
	formantG = formantG[0:cnt]
	'''

	# Crest factor of excitation signal
	excitation = lfilter(A, 1.0, x1)
	extSpec = abs(fft(excitation, 1024))[0:1024 / 2];
	frq = float(fs) / 2.0 * np.array(range(0, 1024 / 2)).astype(float) / (1024. / 2.)
	formantCrest, fbandCrest = get_harmonicPeakiness(extSpec, frq, formantF, formantBw)
	formantCrest = np.append(formantCrest, np.zeros(maxnFormant - formantCrest.shape[0]))

	return formantF, formantBw, formantG, formantCrest, fbandCrest


def get_harmonicPeakiness(extSpec, frq, formantF, formantBw):
	'''
	http://docs.twoears.eu/en/latest/afe/available-processors/spectral-features/
	https://books.google.com/books?id=YSPT1LJqTbIC&pg=PA54&lpg=PA54&dq=spectral+crest&source=bl&ots=oBicAwurSL&sig=P00O8DUtlMlOWgNV5nV2JcMHKbM&hl=en&sa=X&ved=0ahUKEwj6gvz6gNjSAhVC2WMKHceFCwkQ6AEIkAEwFg#v=onepage&q=spectral%20crest&f=false
	'''

	nFormant = sum(formantF != 0)
	formantCrest = np.zeros(nFormant)
	fbandCrest = np.zeros(nFormant)
	# plt.clf(); plt.plot(frq,extSpec)

	# crest factor for each formant
	for n in range(0, nFormant):
		fRange = max(2. * formantBw[n], 100.)
		stFrq = formantF[n] - fRange
		edFrq = formantF[n] + fRange
		# plt.plot([stFrq, stFrq],[min(extSpec), max(extSpec)],'r:'); plt.plot([edFrq, edFrq],[min(extSpec), max(extSpec)],'r:')
		stIdx = max(0, np.argmin(np.abs(frq - stFrq)))
		edIdx = min(np.argmin(np.abs(frq - edFrq)), len(extSpec) - 1)
		formantCrest[n] = max(extSpec[stIdx:edIdx + 1]) / np.mean(extSpec[stIdx:edIdx + 1])
	# plt.clf();plt.plot(extSpec[stIdx:edIdx])

	# crest factor for each frequency band
	nFband = 4
	fbandCrest = np.zeros(nFband)
	freqC1 = [100., 500., 1000., 2000.]
	freqC2 = [500., 1000., 2000., 4000]
	for n in range(0, nFband):
		stIdx = max(0, np.argmin(np.abs(frq - freqC1[n])))
		edIdx = min(np.argmin(np.abs(frq - freqC2[n])), len(extSpec))
		fbandCrest[n] = max(extSpec[stIdx:edIdx]) / np.mean(extSpec[stIdx:edIdx])

	return formantCrest, fbandCrest


def stSpectralCentroidAndSpread_modified(X, fs):
	"""Computes spectral centroid of frame (given abs(FFT))"""
	ind = (np.arange(1, len(X) + 1)) * (fs / (2.0 * len(X)))

	Xt = X.copy()
	if Xt.max() != 0:
		Xt = Xt / Xt.max()
	NUM = np.sum(ind * Xt)
	DEN = np.sum(Xt) + eps

	# Centroid:
	C = (NUM / DEN)

	# Spread:
	S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

	# Normalize:
	C = C / (fs / 2.0)
	S = S / (fs / 2.0)

	return (C, S)



def main():

	wavpath = '../data/EMO-DB/wav/'
	outfile = '../data/testEMO.json'

	fileList = [f for f in os.listdir(wavpath) if f.endswith('.wav')]
	fileList.sort()
	nFile = len(fileList)
	IdTalkerAll = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
	IdTalker = ["" for x in range(nFile)]
	IdText = ["" for x in range(nFile)]
	StrEmo = ["" for x in range(nFile)]
	IdEmo = np.zeros(nFile, int)
	for n in range(0, nFile):
		str = fileList[n]
		IdTalker[n] = str[0:2]
		IdText[n] = str[2:5]
		strE = str[5]
		if strE == 'W':
			StrEmo[n] = 'anger';
			IdEmo[n] = 0
		elif strE == 'L':
			StrEmo[n] = 'boredom';
			IdEmo[n] = 1
		elif strE == 'E':
			StrEmo[n] = 'disgust';
			IdEmo[n] = 2
		elif strE == 'A':
			StrEmo[n] = 'fear';
			IdEmo[n] = 3
		elif strE == 'F':
			StrEmo[n] = 'happiness';
			IdEmo[n] = 4
		elif strE == 'T':
			StrEmo[n] = 'sadness';
			IdEmo[n] = 5
		elif strE == 'N':
			StrEmo[n] = 'neutral';
			IdEmo[n] = 6

	fs = 16000
	winLen = 30 * fs / 1000  # window length (30 msec) in sample
	winStep = 10 * fs / 1000  # window step size (10 msec) in sample
	data = []
	for n in range(0, nFile):
		print n, '/', nFile, ' ', fileList[n]
		[fs, signal] = scipy.io.wavfile.read(wavpath + fileList[n])
		fvec = stFeatureExtraction_modified(signal, fs, winLen, winStep)
		nFrame = fvec.shape[0]
		data.append(
			{'fName': fileList[n], 'IdTalker': IdTalker[n], 'IdText': IdText[n], 'IdEmo': IdEmo[n], 'StrEmo': StrEmo[n],
			 'nFrame': nFrame, 'ftr': fvec.tolist()})

	# data[1]['StrEmo']

	# Save to json file
	with open(outfile, 'w') as fout:
		json.dump(data, fout, separators=(',', ':'), sort_keys=True, indent=0)


if __name__ == "__main__":
	main()
