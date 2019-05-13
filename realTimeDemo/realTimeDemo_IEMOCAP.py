
import os; os.chdir('/Users/johnkim/Google Drive/EmNet/realTimeDemo')
import sys
import numpy as np
import time
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
from pydub import AudioSegment
sys.path.append('../Phase4_IEMOCAP')
import EmNetLib_IEMOCAP as emo
import keras
from keras.models import Sequential, model_from_json
sys.path.append('.')
import featext as ft
import scipy
import pandas as pd
from termcolor import colored

THRESHOLD = 500
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
RATE = 16000
IdTalkerAll = ['F1', 'M1', 'F2', 'M2', 'F3', 'M3', 'F4', 'M4', 'F5', 'M5']
LabelEmotion = ['neu', 'ang', 'sad', 'hap']  # neu, ang, sad, hap (= hap + exc)
TimeForced = 1024  # Forced (either by padding or truncating) number of time frames for each speech file


def is_silent(snd_data):
	"Returns 'True' if below the 'silent' threshold"
	return max(snd_data) < THRESHOLD

def normalize(snd_data):
	"Average the volume out"
	MAXIMUM = 16384
	times = float(MAXIMUM)/max(abs(i) for i in snd_data)

	r = array('h')
	for i in snd_data:
		r.append(int(i*times))
	return r

def trim(snd_data):
	"Trim the blank spots at the start and end"
	def _trim(snd_data):
		snd_started = False
		r = array('h')

		for i in snd_data:
			if not snd_started and abs(i)>THRESHOLD:
				snd_started = True
				r.append(i)

			elif snd_started:
				r.append(i)
		return r

	# Trim to the left
	snd_data = _trim(snd_data)

	# Trim to the right
	snd_data.reverse()
	snd_data = _trim(snd_data)
	snd_data.reverse()
	return snd_data

def add_silence(snd_data, seconds):
	"Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
	r = array('h', [0 for i in xrange(int(seconds*RATE))])
	r.extend(snd_data)
	r.extend([0 for i in xrange(int(seconds*RATE))])
	return r

def record():
	"""
	Record a word or words from the microphone and
	return the data as an array of signed shorts.

	Normalizes the audio, trims silence from the
	start and end, and pads with 0.5 seconds of
	blank sound to make sure VLC et al can play
	it without getting chopped off.
	"""
	time.sleep(0.5)
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT, channels=1, rate=RATE,
		input=True, output=True,
		frames_per_buffer=CHUNK_SIZE)

	num_silent = 0
	snd_started = False

	r = array('h')

	while 1:
		# little endian, signed short
		snd_data = array('h', stream.read(CHUNK_SIZE))
		if byteorder == 'big':
			snd_data.byteswap()
		r.extend(snd_data)

		silent = is_silent(snd_data)

		if silent and snd_started:
			num_silent += 1
		elif not silent and not snd_started:
			snd_started = True

		if snd_started and num_silent > 75:
			break

	sample_width = p.get_sample_size(FORMAT)
	stream.stop_stream()
	stream.close()
	p.terminate()

	r = normalize(r)
	r = trim(r)
	r = add_silence(r, 0.5)
	return sample_width, r

def record_to_file(path):
	"Records from the microphone and outputs the resulting data to 'path'"
	sample_width, data = record()
	data = pack('<' + ('h'*len(data)), *data)

	wf = wave.open(path, 'wb')
	wf.setnchannels(1)
	wf.setsampwidth(sample_width)
	wf.setframerate(RATE)
	wf.writeframes(data)
	wf.close()

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
	'''
	sound is a pydub.AudioSegment
	silence_threshold in dB
	chunk_size in ms

	iterate over chunks until you find the first one with sound
	'''
	trim_ms = 0  # ms

	assert chunk_size > 0  # to avoid infinite loop
	while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
		trim_ms += chunk_size

	return trim_ms

def fvecPreProcessing(fvec):
	MaxHPR = 60
	MinHPR = -60
	iLogEnergy = 1
	iPv = 21
	iPitch = 22
	iFormantF = 23
	iFormantBw = 26
	iFormantG = 29
	iHPP = 32
	iFormantCrest = 37;
	iFBandCrest = 40
	eps = np.finfo(float).eps
	dimXmax = fvec.shape[1]
	fsel = range(0, 8) + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)

	nFrame = fvec.shape[0]
	dimX = len(fsel)
	MaxT = 900
	X = np.zeros((MaxT, dimX))
	Xmean = np.zeros((dimX));
	Xvar = np.zeros((dimX))
	idxVoiced = np.zeros((MaxT), int)
	Zv = np.empty((0, dimX))
	Zu = np.empty((0, dimX))

	Xftr = np.zeros((MaxT, dimXmax))
	idVoiced = [i for i in range(nFrame) if (fvec[i, iPv] > 0.2) & (fvec[i, iLogEnergy] > 37.0)]  # for pyAudioAnalysis
	lenVoiced = len(idVoiced)
	idxVoiced[0:lenVoiced] = np.asarray(idVoiced)
	idU = [i for i in range(0, nFrame) if i not in idVoiced]
	#
	# Feature allocation...
	Xftr[0:nFrame, 0:dimXmax] = np.copy(fvec)
	idx1 = iHPP
	nftr = 4
	idx2 = idx1 + nftr
	Xftr[:, idx1:idx2] = 0
	Xftr[idVoiced, idx1:idx2] = \
		10. * np.log10(np.divide(fvec[idVoiced, iHPP + 1:iHPP + nftr + 1],
								 (fvec[idVoiced, iHPP] + eps).reshape(len(idVoiced), 1)))

	# Clipping
	Xftr[idVoiced, idx1:idx2] = np.clip(Xftr[idVoiced, idx1:idx2], MinHPR, MaxHPR)
	Xftr[idU, idx1:idx2] = MinHPR
	Xftr[idVoiced, iFormantG] = np.clip(Xftr[idVoiced, iFormantG], 0, 125.0)

	# Set unvoiced 0
	Xftr[idU, iPitch] = 0;
	Xftr[idU, iFormantF:iFormantG + 3] = 0;
	Xftr[idU, iFormantCrest:iFormantCrest + 3] = 0;

	X[:, 0:dimX] = np.copy(Xftr[:, fsel])
	return (X, idVoiced)


def main():
	# Load model
	LOSO = 6
	dirModel = '../Phase4_IEMOCAP/model/'
	nameModel = 'conv1lf-conv1gf_tdmode0_modeFsel0_D20T1024-CNN-64x8-mp4-128x3-mp4-LSTM-48_fdrp0.5_bs64_rand1_spkrsi_balanced.json'
	nameWgt = 'conv1lf-conv1gf_tdmode0_modeFsel0_D20T1024-CNN-64x8-mp4-128x3-mp4-LSTM-48_fdrp0.5_bs64_rand1_spkrsi_balanced.hdf5'
	fnameModel = dirModel + 'model_h' + str(LOSO) + '-' + nameModel
	fnameWgt = dirModel + 'weight_h' + str(LOSO) + '-' + nameWgt
	fileModel = open(fnameModel, 'r')
	modelLoadedJson = fileModel.read()
	fileModel.close()
	model = model_from_json(modelLoadedJson)
	model.load_weights(fnameWgt)

	# read in mean/std vector
	xMeanTalker = np.load('xMeanTalker.npy'); xStdTalker = np.load('xStdTalker.npy')

	while True:
		continueOrNo = raw_input("\nEMOTION RECOGNITION? (y/n)")
		if continueOrNo[0] == 'y':
			demoWavName = './recording.wav'
			print colored("\nPLEASE SPEAK IN ANY EMOTION...", 'green')
			time.sleep(1)
			record_to_file(demoWavName)
			sound = AudioSegment.from_file(demoWavName, format='wav')
			start_trim = detect_leading_silence(sound)
			end_trim = detect_leading_silence(sound.reverse())
			duration = len(sound)
			trimmed_sound = sound[start_trim:duration - end_trim]
			trimmed_sound.export(demoWavName, format='wav')
			print colored("DONE. ANALYZING...\n", 'green')

			fs = 16000
			winLen = 30 * fs / 1000  # window length (30 msec) in sample
			winStep = 10 * fs / 1000
			[fs, signal] = scipy.io.wavfile.read(demoWavName)
			fvec = ft.stFeatureExtraction_modified(signal, fs, winLen, winStep)
			X, idVoiced = fvecPreProcessing(fvec)

			# Normalize
			dimUVChange = [14, 15, 16]
			XnormSm = emo.normalize_fmtx(X, xMeanTalker[0], xStdTalker[0])

			XnormSm = np.expand_dims(XnormSm, axis=0)
			XnormSm = emo.smooth(XnormSm, a=0.9)  # smoothing

			# set_uvftrval
			idU = [i for i in range(0, fvec.shape[0]) if i not in idVoiced]
			for k in dimUVChange:
				XnormSm[0, idU, k] = -1

			XnormSmTN = emo.reshape_feature_fixedtime(XnormSm, [fvec.shape[0]], TimeForced)

			# Evaluate
			a = XnormSmTN.shape;
			xTest = np.reshape(XnormSmTN, (a[0], a[1], a[2], 1))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			emotion = model.predict(xTest, verbose=0)[0]
			sortedEmotion = np.argsort(-emotion)

			print("Predicted emotion: ")
			print colored("%s %.2f" % (str(LabelEmotion[sortedEmotion[0]]), emotion[sortedEmotion[0]]) + '%', 'blue')
			print colored("%s %.2f" % (str(LabelEmotion[sortedEmotion[1]]), emotion[sortedEmotion[1]]) + '%', 'blue')
			print colored("%s %.2f" % (str(LabelEmotion[sortedEmotion[2]]), emotion[sortedEmotion[2]]) + '%', 'blue')
			print("\n\n")

			time.sleep(0.5)

		else:
			break


if __name__ == "__main__":
	main()
