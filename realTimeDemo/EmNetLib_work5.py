

'''
build_conv1Dlf_conv1Dgf_lstm()		
Last CNN2D layer bug fixed: border_mode = 'same' --> 'valid':
	model.add(Convolution2D(nFilter[n], lenFilter[n], dimFeat, border_mode='valid', activation='relu'))
'''

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import os
import sys
import platform
sys.path.append("../data")
import itertools

import matplotlib
if platform.system() == 'Darwin':
	import appnope
	appnope.nope()
	matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
import numpy.matlib
import pandas as pd

import scipy
from scipy import stats, signal

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold, LeaveOneGroupOut, GridSearchCV

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, LSTM, Dropout, Flatten, Merge, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Permute, Reshape, BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier, BaseWrapper

import time
import copy

eps = np.finfo(float).eps

IdTalkerAll = [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
LabelEmotion = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral']
LabelText = ['a01', 'a02', 'a04', 'a05', 'a07', 'b01', 'b02', 'b03', 'b09', 'b10']

# start index of feature in ftrVec
iLogEnergy = 1;
iPv = 21;
iPitch = 22;
iFormantF = 23;
iFormantBw = 26;
iFormantG = 29;
iHPP = 32;
iFormantCrest = 37;
iFBandCrest = 40


def summary_dbinfo(Xinfo):

	list(Xinfo)
	#idText = list(set(Xinfo.IdText))

	ns = np.zeros((len(LabelText), len(IdTalkerAll)), 'int')
	for t in range(0, len(LabelText)):
		for h in range(0, len(IdTalkerAll)):
			ns[t,h] = int(Xinfo[ (str(LabelText[t])==Xinfo.IdText) & (IdTalkerAll[h]==Xinfo.IdTalker)
							& (Xinfo.StrEmo=='neutral')].count()[0])


	# Number of samples in neutral emotion (text x talker)
	ns

#def load_emodb_ftr_ext_new(modeFsel, fName='../data/EMO_frameVectorExtended.json'):
def load_emodb_ftr_ext_new(modeFsel, fName='../phase4/EMO_frameVectorExtended_pyAudioAnalysis.json'):
	''' Load and process feature vectors extracted by featext.py
		Feature description:
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
		37-39: crest factor in formant
		40-43: crest factor in 4 frequency bands
	'''

	# _____ Params
	# OffsetHPR = 30	# minimum harmonic peak ratio (used for unvoiced intervals)
	MaxHPR = 60
	MinHPR = -60

	if modeFsel == 0:
		fsel = range(0, 8) + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	# this is the only case raw pitch value is used. otherwise, mel-scaled pitch is used for modeFsel < 20.
	elif modeFsel == 1:
		fsel = range(0, 8) + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 2:
		fsel = range(0, 8) + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                          iHPP + 3) + range(
			iFBandCrest, iFBandCrest + 4)
	#
	elif modeFsel == 3:
		fsel = range(0, 8) + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                          iHPP + 3) + range(
			iFBandCrest, iFBandCrest + 4) + \
		       range(iFormantCrest, iFormantCrest + 2)
	#
	elif modeFsel == 4:
		fsel = [1, 2, 4, 6, 7] + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 5:
		fsel = [1, 2, 4, 6, 7] + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                              iHPP + 3) + range(
			iFBandCrest, iFBandCrest + 4)
	#
	elif modeFsel == 6:
		fsel = [1, 2, 4, 6, 7] + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                              iHPP + 3) + range(
			iFBandCrest, iFBandCrest + 4) + \
		       range(iFormantCrest, iFormantCrest + 2)
	#
	elif modeFsel == 7:
		fsel = [1, 2, 4, 6, 7] + range(9, 17) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 8:
		fsel = [1, 2, 4, 6, 7] + range(9, 17) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                              iHPP + 3) + range(
			iFBandCrest, iFBandCrest + 4)
	#
	elif modeFsel == 9:
		fsel = [1, 2, 4, 6, 7] + range(9, 17) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                              iHPP + 3) + range(
			iFBandCrest, iFBandCrest + 4) + \
		       range(iFormantCrest, iFormantCrest + 2)
	#
	elif modeFsel == 10:
		fsel = range(0, 8) + range(9, 17) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 11:
		fsel = range(0, 8) + range(9, 13) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 12:
		fsel = range(0, 8) + range(9, 12) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 13:
		fsel = range(0, 8) + range(9, 15) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP, iHPP + 3)
	#
	elif modeFsel == 14:
		fsel = [0, 1, 2, 3, 4, 5, 7] + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + [iFormantG] + range(iHPP,
		                                                                                                    iHPP + 3)
	elif modeFsel == 20:
		fsel = range(0, 8) + range(9, 14) + [iPv] + [iPitch] + [iFormantBw] + range(iHPP, iHPP + 3)
	# remove spectral flux
	dimX = len(fsel)

	# _____ Load & process data to get feature vector for static classification
	# df0 = pd.read_json('../data/EMO_frameVector.json')
	df = pd.read_json(fName)  # DataFrame

	nFile = df.shape[0]
	Xinfo = df.loc[:, ['fName', 'IdTalker', 'IdText', 'IdEmo', 'StrEmo', 'nFrame']]
	# MaxT = max(Xinfo['nFrame'])	# maximum number of frames for each file
	MaxT = 900
	dimXmax = np.array(df.ix[0]['ftr']).shape[1]

	X = np.zeros((nFile, MaxT, dimX))
	Xmean = np.zeros((nFile, dimX));
	Xvar = np.zeros((nFile, dimX))
	lenVoiced = np.zeros(nFile, int)
	idxVoiced = np.zeros((nFile, MaxT), int)
	Zv = np.empty((0, dimX))
	Zu = np.empty((0, dimX))
	for n in range(0, nFile):
		Xftr = np.zeros((MaxT, dimXmax))
		nFrame = df.ix[n]['nFrame']
		ftr = np.array(df.ix[n]['ftr'])
		# idVoiced = [i for i in range(nFrame) if (ftr[i,iPv]>0.3) & (ftr[i,iLogEnergy]>35.0)]
		# idVoiced = [i for i in range(nFrame) if (ftr[i, iPv] > 0.5) & (ftr[i, iLogEnergy] > 37.0)]
		# idVoiced = [i for i in range(nFrame) if (ftr[i, iPv] > 0.4) & (ftr[i, iLogEnergy] > 37.0)]  # for pyAudioAnalysis
		idVoiced = [i for i in range(nFrame) if (ftr[i, iPv] > 0.2) & (ftr[i, iLogEnergy] > 37.0)]

		lenVoiced[n] = len(idVoiced)
		idxVoiced[n, 0:lenVoiced[n]] = np.asarray(idVoiced)
		idU = [i for i in range(0, nFrame) if i not in idVoiced]
		#
		# Feature allocation...
		Xftr[0:nFrame, 0:dimXmax] = np.copy(ftr)
		if modeFsel != 0 or modeFsel >= 20:
			Xftr[:nFrame, iPitch] = hz2mel(Xftr[:nFrame, iPitch])
		idx1 = iHPP
		nftr = 4;
		idx2 = idx1 + nftr
		Xftr[:, idx1:idx2] = 0
		Xftr[idVoiced, idx1:idx2] = \
			10. * np.log10(np.divide(ftr[idVoiced, iHPP + 1:iHPP + nftr + 1],
			                         (ftr[idVoiced, iHPP] + eps).reshape(len(idVoiced), 1)))

		# Clipping
		Xftr[idVoiced, idx1:idx2] = np.clip(Xftr[idVoiced, idx1:idx2], MinHPR, MaxHPR)
		Xftr[idU, idx1:idx2] = MinHPR
		Xftr[idVoiced, iFormantG] = np.clip(Xftr[idVoiced, iFormantG], 0, 125.0)

		# Set unvoiced 0
		Xftr[idU, iPitch] = 0;
		Xftr[idU, iFormantF:iFormantG + 3] = 0;
		Xftr[idU, iFormantCrest:iFormantCrest + 3] = 0;

		# for histogram view
		#Zv = np.append(Zv, (Xftr[idVoiced,:])[:,fsel], axis=0)
		#Zu = np.append(Zu, (Xftr[idU, :])[:,fsel], axis=0)

		X[n, :, 0:dimX] = np.copy(Xftr[:, fsel])

		# compute per-file-{mean, var} from voiced intervals only
		Xmean[n, 0:dimX] = np.mean(X[n, idVoiced, :], axis=0)
		Xvar[n, 0:dimX] = np.var(X[n, idVoiced, :], axis=0)

	Xtarget = np.array(Xinfo.IdEmo)

	#k=16
	#z = Zv[:, k]
	#plt.clf()
	#n, bins, patches = plt.hist(z, 50, normed=0, facecolor='blue', alpha=0.5)

	# _____ Append info to Xinfo
	Xinfo = Xinfo.assign(lenVoiced=lenVoiced)
	z = []
	for n in range(0, nFile):
		z.append([idxVoiced[n, 0:lenVoiced[n]]])
	Xinfo['idVoiced'] = z
	z = []
	for n in range(0, nFile):
		z.append([Xmean[n, 0:dimX]])
	Xinfo['Xmean'] = z
	z = []
	for n in range(0, nFile):
		z.append([Xvar[n, 0:dimX]])
	Xinfo['Xvar'] = z

	return (X, Xtarget, Xinfo)


def set_uvftrval(X, Xinfo, idxChange):

	""" This makes performance worse! """
	nFile = (X.shape)[0]
	for n in range(0, nFile):
		idU = [i for i in range(0, Xinfo.nFrame[n]) if i not in np.concatenate(Xinfo.idVoiced[n])]
		for k in idxChange:
			X[n, idU, k] = -1
	return X


def feature_normalization(X, Xinfo, idxLOSO, modeSpkrdep='sdall'):
	"""
	Feature normalization using mean and std vector.
	"""
	nFile = (X.shape)[0]
	dimTime = (X.shape)[1]
	dimFeat = (X.shape)[2]

	xMeanTalker = np.zeros((len(IdTalkerAll), dimFeat))
	xStdTalker = np.ones((len(IdTalkerAll), dimFeat))
	xstaticMeanTalker = np.zeros((len(IdTalkerAll), dimFeat*2))
	xstaticStdTalker = np.ones((len(IdTalkerAll), dimFeat*2))

	Xnorm = np.zeros((nFile, (X.shape)[1], dimFeat))
	XnormStatic = np.zeros((nFile, dimFeat * 2))

	# Per-file mean and var vector
	xmean = np.concatenate(np.array(Xinfo['Xmean']))  # concatenate: unwrapping object
	xvar = np.concatenate(np.array(Xinfo['Xvar']))

	# Standardizing vectors
	if modeSpkrdep == 'si':
		print '### speaker-independent mode: '
		print ' Vtr = E{txt_all, emotion_all, spkr_tr}, Vtst = Vtr'
		idx = (np.where((Xinfo['IdTalker'] != IdTalkerAll[idxLOSO])))[0]
		# for EmNet: zm = sample mean of temporal mean & zs = sample mean of temporal std
		zm = np.mean(xmean[idx,:],axis=0)
		zs = np.sqrt(np.mean(xvar[idx, :], axis=0))
		xMeanTalker = np.matlib.repmat(zm, len(IdTalkerAll), 1)
		xStdTalker = np.matlib.repmat(zs, len(IdTalkerAll), 1)
		# for SVM
		xstaticMeanTalker = np.mean(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)
		xstaticStdTalker = np.std(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)
		xstaticMeanTalker = np.matlib.repmat(xstaticMeanTalker, len(IdTalkerAll), 1)
		xstaticStdTalker = np.matlib.repmat(xstaticStdTalker, len(IdTalkerAll), 1)

	elif modeSpkrdep == 'sdall':
		print '### speaker-dependent, with all emotions'
		print ' Vtr = E{txt_all, emotion_all | spkr_tr}, Vtst = E{txt_all, emotion_all | spkr_tst}'
		for h in range(0, len(IdTalkerAll)):
			# for EmNet
			idx = (np.where((Xinfo['IdTalker'] == IdTalkerAll[h])))[0]
			xMeanTalker[h] = np.mean(xmean[idx, :], axis=0)
			xStdTalker[h] = np.sqrt(np.mean(xvar[idx, :], axis=0))
			# for SVM
			xstaticMeanTalker[h] = np.mean(np.hstack((xmean[idx,:], np.sqrt(xvar[idx,:]))), axis=0)
			xstaticStdTalker[h] = np.std(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)

	elif modeSpkrdep == 'sdall-sdn':
		print '### speaker-dependent, with all emotions'
		print ' Vtr = E{txt_all, emotion_all | spkr_tr}, Vtst = E{txt_all, neutral | spkr_tst}'
		for h in range(0, len(IdTalkerAll)):
			if h == idxLOSO:
				idx = np.where(((Xinfo['IdTalker'] == IdTalkerAll[h]) & (Xinfo['StrEmo'] == 'neutral')))[0]
			else:
				idx = (np.where((Xinfo['IdTalker'] == IdTalkerAll[h])))[0]
			# for EmNet
			xMeanTalker[h] = np.mean(xmean[idx, :], axis=0)
			xStdTalker[h] = np.sqrt(np.mean(xvar[idx, :], axis=0))
			# for SVM
			xstaticMeanTalker[h] = np.mean(np.hstack((xmean[idx,:], np.sqrt(xvar[idx,:]))), axis=0)
			xstaticStdTalker[h] = np.std(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)

	elif modeSpkrdep == 'sdall-sdn1':
		print '### speaker-dependent, with neutral emotion and only one common text'
		print ' Vtr = E{txt_all, emotion_all | spkr_tr}, Vtst = E{txt_1, neutral | spkr_tst}'
		for h in range(0, len(IdTalkerAll)):
			if h == idxLOSO:
				idx = np.where(((Xinfo['IdTalker'] == IdTalkerAll[h]) & (Xinfo['StrEmo'] == 'neutral')
				                & (Xinfo['IdText'] == LabelText[0])))[0]  # a01 only
				print '# h = ' + str(h)
				print idx
			else:
				idx = (np.where((Xinfo['IdTalker'] == IdTalkerAll[h])))[0]
			# for EmNet
			xMeanTalker[h] = np.mean(xmean[idx, :], axis=0)
			xStdTalker[h] = np.sqrt(np.mean(xvar[idx, :], axis=0))
			# for SVM
			xstaticMeanTalker[h] = np.mean(np.hstack((xmean[idx,:], np.sqrt(xvar[idx,:]))), axis=0)
			xstaticStdTalker[h] = np.std(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)

	elif modeSpkrdep == 'sdn':
		print '### speaker-dependent, with neutral emotion'
		print ' Vtr = E{txt_all, neutral | spkr_tr}, Vtst = E{txt_1, neutral | spkr_tst}'
		for h in range(0, len(IdTalkerAll)):
			idx = np.where(((Xinfo['IdTalker'] == IdTalkerAll[h]) & (Xinfo['StrEmo'] == 'neutral')))[0]
			# for EmNet
			xMeanTalker[h] = np.mean(xmean[idx, :], axis=0)
			xStdTalker[h] = np.sqrt(np.mean(xvar[idx, :], axis=0))
			# for SVM
			xstaticMeanTalker[h] = np.mean(np.hstack((xmean[idx,:], np.sqrt(xvar[idx,:]))), axis=0)
			xstaticStdTalker[h] = np.std(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)

	elif modeSpkrdep == 'sdn1':
		print '### speaker-dependent, with neutral emotion and only one common text'
		print ' Vtr = E{txt_1, neutral | spkr_tr}, Vtst = E{txt_1, neutral | spkr_tst}'
		for h in range(0, len(IdTalkerAll)):
			idx = np.where(((Xinfo['IdTalker'] == IdTalkerAll[h]) & (Xinfo['StrEmo'] == 'neutral')
		                & (Xinfo['IdText'] == LabelText[0])))[0]  # a01 only
			# for EmNet
			xMeanTalker[h] = np.mean(xmean[idx, :], axis=0)
			xStdTalker[h] = np.sqrt(np.mean(xvar[idx, :], axis=0))
			# for SVM
			xstaticMeanTalker[h] = np.mean(np.hstack((xmean[idx,:], np.sqrt(xvar[idx,:]))), axis=0)
			xstaticStdTalker[h] = np.std(np.hstack((xmean[idx, :], np.sqrt(xvar[idx, :]))), axis=0)
	else:
		print 'spkrdep option error'
		raise SystemExit

	# Normalization
	for h in range(0, len(IdTalkerAll)):
		idx = (np.where((Xinfo['IdTalker'] == IdTalkerAll[h])))[0]
		for k in range(0, len(idx)):
			n = idx[k]
			nf = Xinfo.nFrame[n]
			Xnorm[n, 0:Xinfo.nFrame[n], 0:dimFeat] = normalize_fmtx(X[n, 0:Xinfo.nFrame[n], 0:dimFeat], xMeanTalker[h], xStdTalker[h])
		print h, np.max(Xnorm[n,:,:])
		# This part is for producing static feature - svm
		xstatic = np.hstack((xmean[idx,:], np.sqrt(xvar[idx,:])))
		XnormStatic[idx, 0:dimFeat * 2] = normalize_fmtx(xstatic, xstaticMeanTalker[h], xstaticStdTalker[h])

	return (Xnorm, XnormStatic)


def get_tdfeature(X, Xinfo):

	# Time-derivative
	nFile = (X.shape)[0]
	dimTime = (X.shape)[1]
	dimFeat = (X.shape)[2]
	dimFeatOut = dimFeat * 2
	Xnew = np.zeros((nFile, dimTime, dimFeatOut))

	for n in range(0, nFile):
		df = Xinfo.ix[n]
		idVoiced = np.concatenate(df.idVoiced)
		idU = [i for i in range(0, Xinfo.nFrame[n]) if i not in idVoiced]

		# gradient is affected by forced unvoiced values in load_emodb_ftr_ext_new. this should be enhanced.
		z = np.concatenate(np.gradient(X[n, 0:dimTime, 0:dimFeat]))[0:dimTime, 0:dimFeat]

		# set z[ idU[end-1:end+1] ] to be zero (last two frames in idU plus one more frame)
		id = idU[-2:]
		id.append(idU[-1] + 1)
		z[idU, :] = 0;
		z[id, :] = 0
		Xnew[n, 0:dimTime, 0:dimFeat] = np.copy(X[n, 0:dimTime, 0:dimFeat])
		Xnew[n, 0:dimTime, dimFeat:dimFeatOut] = np.copy(z)

	return Xnew

def normalize_fmtx(x, xmean, xstd):
	nRow = (x.shape)[0]
	xmeanRep = np.matlib.repmat(xmean, nRow, 1)
	xstdRep = np.matlib.repmat(xstd, nRow, 1)
	xnew = np.divide((x - xmeanRep), 7.*xstdRep)
	return(xnew)

def smooth(x, a=0.8):
	hx = x.shape[0] #files
	mx = x.shape[1] #time
	nx = x.shape[2]
	
	y = np.zeros((hx, mx, nx))
	for h in range(0, hx):
		y[h,0] = np.copy(x[h,0,:])
		for m in range(1, mx):
			y[h,m] = a * x[h,m] + (1. - a)*y[h,m-1]
	return(y)

def hz2mel(f):
	"""Convert an array of frequency in Hz into mel."""
	return 1127.01048 * np.log(f/700. +1)

def reshape_feature_fixedtime(xin, nFrame, lenForced):
	''' Make the length of time (2nd dim of xin) to lenForced.
		Pad leading zeros if the length of xin in less than lenForced. Otherwise, truncate it.
	'''	
	nFile = (xin.shape)[0]
	dimFeat = (xin.shape)[2]
	xout = np.zeros((nFile, lenForced, dimFeat))
	for i in range(0, nFile):
		#print i, ' out of ', dim0
		if nFrame[i]  < lenForced:
			lenPad = lenForced - nFrame[i]
			a = np.concatenate((np.zeros((lenPad, dimFeat)), xin[i, 0:nFrame[i], :]))
		elif nFrame[i]  >= lenForced:
			a = xin[i,  0:lenForced, :]
			
		xout[i,0:lenForced,0:dimFeat] = a

	return(xout)

def run_svm(training_data, test_data, c, randSeed):
	np.random.seed(randSeed)
	
	clf = svm.SVC(C=c,random_state=randSeed)
	print 'Test2'
	#clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(training_data[0], training_data[1])
	
	predictionTrain = [int(a) for a in clf.predict(training_data[0])]	
	predictionTest = [int(a) for a in clf.predict(test_data[0])]
	ncorrectTrain = sum(int(a == y) for a, y in zip(predictionTrain, training_data[1]))
	ncorrectTest = sum(int(a == y) for a, y in zip(predictionTest, test_data[1]))
	
	rateTrain = float(ncorrectTrain)/float(len(training_data[1]))
	rateTest = float(ncorrectTest)/float(len(test_data[1]))
	
	#print "Baseline classifier using an SVM."
	#print "%.2f%%: %s of %s values correct." % (100.*float(num_correct)/float(len(test_data[1])), num_correct, len(test_data[1]))
	return(rateTrain, rateTest, predictionTrain, predictionTest)



def build_conv1Dgf_lstm(dimTime, dimFeat,
				nFilter, lenFilter, lenPool, nLSTM, fDropout, nClass=7):
	''' Conv1dGF -...-Conv1dGF-LSTM-...LSTM-Dense 	
		Build a network consisting of (multiple) Conv1D with global filters (dimFeat x lenFilter),
		where each global filter spans across all feature dimension, followed by (multiple) LSTM
			or
		LSTM-...-LSTM-Dense
	'''
	model = Sequential()
	nLcnn = len(nFilter)
	nLlstm = len(nLSTM)
	if nLcnn:
		print nFilter[0], lenFilter[0], lenPool[0]

		model.add(Conv1D(nFilter[0], lenFilter[0], input_shape = (dimTime, dimFeat), padding='same', activation='relu'))
		model.add(MaxPooling1D(pool_size=lenPool[0]))

		for n in range(1,nLcnn):
			#tf
			model.add(Conv1D(nFilter[n], lenFilter[n], padding='same', activation='relu'))
			model.add(MaxPooling1D(pool_size=lenPool[n]))

		for n in range(0,nLlstm-1):
			model.add(LSTM(nLSTM[n], return_sequences=True))
			model.add(Dropout(fDropout))
		model.add(LSTM(nLSTM[nLlstm-1]))
		model.add(Dropout(fDropout))

	else:	# LSTM only
		if nLlstm==1:
			model.add(LSTM(nLSTM[0], input_shape = (dimTime, dimFeat)))
			model.add(Dropout(fDropout))		
		else:
			model.add(LSTM(nLSTM[0], input_shape = (dimTime, dimFeat), return_sequences=True))			
			model.add(Dropout(fDropout))	

			for n in range(1,nLlstm-1):
				model.add(LSTM(nLSTM[n], return_sequences=True))
				model.add(Dropout(fDropout))
			model.add(LSTM(nLSTM[nLlstm-1]))
			model.add(Dropout(fDropout))
					
	model.add(Dense(nClass, activation='softmax'))	

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print 'conv1Dgf_lstm: CNN = ', lenFilter, lenPool, ', LSTM = ', nLSTM
	print(model.summary())
	return model

def build_conv1Dlf_conv1Dgf_lstm(dimTime, dimFeat,
				nFilter, lenFilter, lenPool, nLSTM, fDropout, nClass=7):
	''' Conv1dLF -...-Conv1dLF-Conv1dGF-LSTM-...LSTM-Dense 	
		Build a network consisting of (multiple) Conv1D with local filters (1 x lenFilter),
		where each local filter is 1-dim and spans along time dimension,
		followed by Conv1D with global filter (dimFeat x lenFilter) and (multiple) LSTM
	'''
	model = Sequential()
	nLcnn = len(nFilter)
	nLlstm = len(nLSTM)
	if nLcnn:
		# First CNN2D - effectively it is 1D-convolution, but keras Conv1D doesn't support input dimension bigger than 1
		# TF defaul format = channels_last
		if keras.backend._image_data_format == 'channels_last':
			model.add(Conv2D(nFilter[0], (lenFilter[0], 1), input_shape=(dimTime, dimFeat, 1), padding='same', activation='relu'))
		elif keras.backend._image_data_format == 'channels_first':
			model.add(Conv2D(nFilter[0], (lenFilter[0], 1), input_shape=(1, dimTime, dimFeat), padding='same', activation='relu'))

		model.add(MaxPooling2D(pool_size=(lenPool[0],1)))

		for n in range(1,nLcnn-1):
			# tf with keras 2.0.9
			model.add(Conv2D(nFilter[n], (lenFilter[n], 1), padding='same', activation='relu'))
			model.add(MaxPooling2D(pool_size=(lenPool[0], 1)))

		# Last CNN2D layer
		n = nLcnn - 1
		model.add(Conv2D(nFilter[n], (lenFilter[n], dimFeat), padding='valid', activation='relu'))
		model.add(MaxPooling2D(pool_size=(lenPool[n], 1)))

		# Re-ordering & reshaping to connect LSTM, where the format of input to LSTM should be (None, timestep, dimfeat)
		if keras.backend._image_data_format == 'channels_last':
			model.add(Permute((1, 3, 2)))
		elif keras.backend._image_data_format == 'channels_first':
			model.add(Permute((2, 1, 3)))

		a=model.output_shape
		model.add(Reshape((a[1], a[2]*a[3])))

		for n in range(0,nLlstm-1):
			model.add(LSTM(nLSTM[n], return_sequences=True))
			model.add(Dropout(fDropout))
		model.add(LSTM(nLSTM[nLlstm-1]))
		model.add(Dropout(fDropout))

	else:
		sys.exit('Provide a proper option.')
					
	model.add(Dense(nClass, activation='softmax'))	

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print 'conv1Dlf_conv1Dgf_lstm: CNN = ', lenFilter, lenPool, ', LSTM = ', nLSTM
	print(model.summary())
	return model

def build_conv2D_conv1Dgf_lstm(dimTime, dimFeat,
				nFilter, lenFilter, lenPool, nLSTM, fDropout, nClass=7):
	''' Conv2d -...-Conv2d-Conv1dGF-LSTM-...LSTM-Dense 	
		Build a network consisting of (multiple) Conv2D with filters (lenFilter x lenFilter),
		followed by Conv1D with global filter (dimFeat x lenFilter) and (multiple) LSTM
	''' 
	model = Sequential()
	nLcnn = len(nFilter)
	nLlstm = len(nLSTM)
	if nLcnn:
		# First CNN2D
		# th
		#model.add(Convolution2D(nFilter[0], lenFilter[0], lenFilter[0], input_shape = (1, dimTime, dimFeat), border_mode='same', activation='relu'))
		#model.add(MaxPooling2D(pool_size=(lenPool[0],lenPool[0])))
		# tf with keras 2.0.9
		if keras.backend._image_data_format == 'channels_last':
			model.add(Conv2D(nFilter[0], (lenFilter[0], lenFilter[0]), input_shape=(dimTime, dimFeat, 1), padding='same', activation='relu'))
		elif keras.backend._image_data_format == 'channels_first':
			model.add(Conv2D(nFilter[0], (lenFilter[0], lenFilter[0]), input_shape=(1, dimTime, dimFeat), padding='same', activation='relu'))

		model.add(MaxPooling2D(pool_size=(lenPool[0], lenPool[0])))
		
		for n in range(1,nLcnn-1):
			model.add(Convolution2D(nFilter[n], (lenFilter[n], lenFilter[n]), padding='same', activation='relu'))
			model.add(MaxPooling2D(pool_size=(lenPool[n],lenPool[n])))

		# Last CNN2D layer
		n = nLcnn - 1
		model.add(Conv2D(nFilter[n], lenFilter[n], dimFeat, border_mode='same', activation='relu'))
		model.add(MaxPooling2D(pool_size=(lenPool[n],1)))

		# Re-ordering & reshaping to connect LSTM, where the format of input to LSTM should be (None, timestep, dimfeat)
		if keras.backend._image_data_format == 'channels_last':
			model.add(Permute((1, 3, 2)))
		elif keras.backend._image_data_format == 'channels_first':
			model.add(Permute((2, 1, 3)))
		
		a=model.output_shape
		model.add(Reshape((a[1], a[2]*a[3])))

		for n in range(0,nLlstm-1):
			model.add(LSTM(nLSTM[n], return_sequences=True))
			model.add(Dropout(fDropout))
		model.add(LSTM(nLSTM[nLlstm-1]))
		model.add(Dropout(fDropout))

	else:
		sys.exit('Provide a proper option.')
					
	model.add(Dense(nClass, activation='softmax'))	

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print 'conv2D_conv1Dgf_lstm: CNN = ', lenFilter, lenPool, ', LSTM = ', nLSTM
	print(model.summary())
	return model

def build_conv2D(dimTime, dimFeat,
				nFilter, lenFilter, lenPool, fDropout, nClass=7):
	''' Conv2d -...-Conv2d-Dense 	
		Build a network consisting of (multiple) Conv2D with filters (lenFilter x lenFilter),
	''' 
	model = Sequential()
	nLcnn = len(nFilter)
	if nLcnn:
		# First CNN2D
		# th
		# model.add(Convolution2D(nFilter[0], lenFilter[0], lenFilter[0], input_shape = (1, dimTime, dimFeat), border_mode='same', activation='relu'))
		# model.add(MaxPooling2D(pool_size=(lenPool[0],lenPool[0])))
		# tf
		if keras.backend._image_data_format == 'channels_last':
			model.add(Conv2D(nFilter[0], (lenFilter[0], lenFilter[0]), input_shape=(dimTime, dimFeat, 1), padding='same', activation='relu'))
		elif keras.backend._image_data_format == 'channels_first':
			model.add(Conv2D(nFilter[0], (lenFilter[0], lenFilter[0]), input_shape=(1, dimTime, dimFeat), padding='same', activation='relu'))

		model.add(MaxPooling2D(pool_size=(lenPool[0], lenPool[0])))

		for n in range(1,nLcnn):
			# model.add(Convolution2D(nFilter[n], lenFilter[n], lenFilter[n], border_mode='same', activation='relu'))
			# model.add(MaxPooling2D(pool_size=(lenPool[n],lenPool[0])))
			model.add(Convolution2D(nFilter[n], (lenFilter[n], lenFilter[n]), padding='same', activation='relu'))
			model.add(MaxPooling2D(pool_size=(lenPool[n],lenPool[n])))

		model.add(Flatten())
	else:
		sys.exit('This is for cnn2D-LSTM only. Provide a proper option.')
					
	model.add(Dense(nClass, activation='softmax'))	

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print 'conv2D: CNN = ', lenFilter, lenPool
	print(model.summary())
	return model


	
def build_modular_model(dimTime, dimFeat,
				nFilter, lenFilter, lenPool, nLSTM, fDropout, nClass=7):
	'''
	This needs further works
	'''	
	if dimFeat != 20:
		print 'dimFeat =', str(dimFeat), ': Not supported yet'
		sys.exit()
			
	nLcnn = len(nFilter)
	nLlstm = len(nLSTM)
	if nLcnn:
		print nFilter[0], lenFilter[0], lenPool[0]
		pm = [Sequential() for _ in range(0, dimFeat)]		
		for i in range(0,dimFeat):
			pm[i].add(Convolution1D(nb_filter=nFilter[0], filter_length=lenFilter[0], input_shape = (dimTime, 1), border_mode='same', activation='relu'))
			pm[i].add(MaxPooling1D(pool_length=lenPool[0]))

		model = Sequential()
		model.add(Merge([pm[0],pm[1],pm[2],pm[3],pm[4],pm[5],pm[6],pm[7],pm[8],pm[9],pm[10],pm[11],pm[12],pm[13],pm[14],pm[15],pm[16],pm[17],pm[18],pm[19]], mode='concat'))

		for n in range(0,nLlstm-1):
			model.add(LSTM(nLSTM[n], return_sequences=True))
			model.add(Dropout(fDropout))
		model.add(LSTM(nLSTM[nLlstm-1]))
		model.add(Dropout(fDropout))
					
	model.add(Dense(nClass, activation='softmax'))	

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	# This function prints and plots the confusion matrix.
	# Normalization can be applied by setting `normalize=True`.
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	print thresh
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
		         horizontalalignment="center",
		         color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')



'''	
def normalize_fmtx2(x, xmean, xstd):
	nRow = (x.shape)[0]
	nCol = (x.shape)[1]
	xmeanRep = np.matlib.repmat(xmean, nRow, 1)
	xstdRep = np.matlib.repmat(xstd, nRow, 1)
	#xnew = np.divide(x, 6.*xstd) + 0.5*np.ones((nRow,nCol),float)
	xnew = np.divide((x - xmeanRep), 6.*xstdRep) + 0.5*np.ones((nRow,nCol),float)
	return(xnew)
	
def feature_normalization_talker2(X, Xinfo, modeTD=0):

	IdTalkerAll = [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
	nFile = (X.shape)[0]
	dimTime = (X.shape)[1]
	dimFeat = (X.shape)[2]
	meanMeanXtalker = np.zeros((len(IdTalkerAll),dimFeat))
	meanStdXtalker = np.ones((len(IdTalkerAll),dimFeat))
	dimFeatOut = dimFeat
	if modeTD:
		dimFeatOut = dimFeat*2		
	Xnorm = np.zeros((nFile, (X.shape)[1], dimFeatOut))
	XnormStatic = np.zeros((nFile, dimFeatOut*2))
	
	# Per-file mean and var vector	
	xmean = np.concatenate(np.array(Xinfo['Xmean']))	# concatenate: unwrapping object
	xvar = np.concatenate(np.array(Xinfo['Xvar']))

	np.max(X)
			
	for h in range(0, len(IdTalkerAll)):
		
		# Talker mean and std
		idx = (np.where((Xinfo['IdTalker']==IdTalkerAll[h])))[0]	# caution. tailing [0]
		xm = np.array([])
		xv = np.array([])
		for k in range(0, len(idx)):
			n = idx[k]
			xm = np.vstack((xm, xmean[n,:])) if xm.size else xmean[n,:]
			xv = np.vstack((xv, xvar[n,:])) if xv.size else xvar[n,:]
		
		# This part is for producing static feature - svm
		zstatic = np.hstack((xm, np.sqrt(xv)))	
		zm = np.mean(zstatic, axis=0)
		zs = np.std(zstatic, axis=0)
		zstaticNorm = normalize_fmtx(zstatic, zm, zs)
		XnormStatic[idx,0:dimFeat*2] = zstaticNorm
		
		meanMeanXtalker[h] = np.mean(xm, axis=0)
		meanStdXtalker[h] = np.sqrt(np.mean(xv, axis=0))
		
		# Normalization by mean and std
		for k in range(0, len(idx)):
			n = idx[k]
			Xnorm[n,0:Xinfo.nFrame[n],0:dimFeat] = normalize_fmtx2(X[n, 0:Xinfo.nFrame[n],0:dimFeat], meanMeanXtalker[h], meanStdXtalker[h])
			
			# Set values in unvoiced intervals zero
			df = Xinfo.ix[n]
			idVoiced = np.concatenate(df.idVoiced)				
			idU = [i for i in range(0,Xinfo.nFrame[n]) if i not in idVoiced]
			Xnorm[n,idU,:] = 0

	return(Xnorm, XnormStatic)	

		
'''
