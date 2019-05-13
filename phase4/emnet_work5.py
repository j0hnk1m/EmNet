'''
EmNet Training
'''


# See https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# imports
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

import EmNetLib_work5 as emo

from collections import Counter
from optparse import OptionParser
import argparse

def custom_get_params(self, **params):
	res = copy.deepcopy(self.sk_params)
	res.update({'build_fn': self.build_fn})
	return res

def main():

	parser = argparse.ArgumentParser(description='EmNet Training.', epilog='This is epilog.')
	# Mandatory parameters
	parser.add_argument('-m', dest='model', type=str, required=True, help='[svm | lstm | conv1gf | conv1lf-conv1gf | conv2-conv1gf | conv2]')
	parser.add_argument('-ftrmode', dest='mode_ftr', type=int, help='Feature selection mode')	
	parser.add_argument('-tdmode', dest='mode_td', type=int, 
	                    help='Time-derivative mode, to enable appending time-derivative feature vector: 0 or 1')	
	parser.add_argument('-cnn', dest='varCNN', nargs='+', type=int, 
					help='Convolutional layer parameters [nFilter lenFilter lenPool]')
	parser.add_argument('-lstm', dest='varLSTM', nargs='+', type=int, help='LSTM parameters [nLSTM_1 nLSTM_2 ... ]')
	
	# Optional parameters
	parser.add_argument('-fdrop', dest='fDropout', type=float, default='0.25', help='Dropout factor [0.25]')
	parser.add_argument('-bsize', dest='batchSize', type=int, default='128', help='Batch size: [128] or n')
	parser.add_argument('-epoch', dest='epochLen', type=int, default='250', help='Epoch length: [250] or n')
	parser.add_argument('-save', dest='flagSaveModel', type=int, default='0', help='Save model flag: [0] or 1')
	parser.add_argument('-randseed', dest='randSeed', type=int, default='2', help='Random number seed [2] or n')
	parser.add_argument('-dir', dest='dirOut', type=str, default='.', help='Path to save output files [./]')
	parser.add_argument('-spkrdep', dest='modeSpkrdep', type=str, default='sdall',
	                    help='si = speaker-independent feature normalization, \n' +
	                         'sdall = speaker-dependent (using all texts and emotions), \n' +
	                         'sdn = speaker-dependent using all neutral texts, \n' +
	                         'sdn1 = speaker-dependent using 1 common neutral text'
	                         'sdall-sdn = training in sdall and testing in sdn'
	                         'sdall-sdn1 = training in sdall and testing in sdn1'
	                    )

	args = parser.parse_args()
	if args.mode_ftr is None or args.mode_td is None:
		sys.exit('Error: ftrmode and tdmode are missing.')
		
	strModel = str(args.model)
	modeFsel = int(args.mode_ftr)
	modeTD = int(args.mode_td)
	fDropout = float(args.fDropout) 
	batchSize = int(args.batchSize)
	epochLen = int(args.epochLen)
	flagSaveModel = int(args.flagSaveModel)
	randSeed = int(args.randSeed)
	dirOut = str(args.dirOut)
	modeSpkrdep = str(args.modeSpkrdep)

	if os.path.exists(dirOut) == False:
		os.makedirs(dirOut)

	# Parsing arguments
	nLayerCNN = 0; nFilter=[]; lenFilter=[]; lenPool=[]
	nLayerLSTM = 0; nLSTM=[]	
	print '\nMODEL: '+strModel+' with modeFsel='+str(modeFsel)+', modeTd='+str(modeTD)	
	if strModel != 'svm' and strModel != 'lstm':		# CNN	
		if args.varCNN is None:
			sys.exit('Error: CNN parameters are missing...')
		else:
			varCNN = np.array(args.varCNN)
			nLayerCNN = len(varCNN)/3
			nFilter = varCNN[::3]
			lenFilter = varCNN[1::3]
			lenPool = varCNN[2::3]
			
			msg = '  CNN: '
			for i in range(0,nLayerCNN):
				msg += '-' +str(nFilter[i])+'x'+str(lenFilter[i])+'mp'+str(lenPool[i])
			print msg
				
		if args.varLSTM is not None:
			varLSTM = np.array(args.varLSTM)
			nLayerLSTM = len(varLSTM)	
			nLSTM = varLSTM
			msg = '  LSTM: '
			for i in range(0,nLayerLSTM):
				msg += '-' +str(nLSTM[i])
			print msg
			
	elif strModel == 'lstm':		# LSTM only
		if args.varLSTM is None:
			sys.exit('Error: LSTM parameters are missing...')
		else:
			varLSTM = np.array(args.varLSTM)
			nLayerLSTM = len(varLSTM)	
			nLSTM = varLSTM	
			msg = '  LSTM: '
			for i in range(0,nLayerLSTM):
				msg += '-' +str(nLSTM[i])
			print msg

	print ' fDropout='+str(fDropout)+', batchSize='+str(batchSize)+', randseed='+str(randSeed)+', saveModel='+str(flagSaveModel)
	BaseWrapper.get_params = custom_get_params
	
	'''
	for debugging

	#
	strModel = 'conv1lf-conv1gf'; modeFsel = 0; modeTD = 0; fDropout = 0.25; batchSize = 64; epochLen = 128; flagSaveModel = 0;
	randSeed = 2;
	nLayerCNN = 2; nLayerLSTM = 1;
	nFilter = [64, 128]; lenFilter = [6, 2]; lenPool = [4, 2];
	nLSTM = [48];
	modeSpkrdep = 'sdall-sdn1'

	#
	strModel = 'conv1gf'; modeFsel = 0; modeTD = 0; fDropout = 0.25; batchSize = 64; epochLen = 128; flagSaveModel = 0;
	randSeed = 2;
	nLayerCNN = 1; nLayerLSTM = 2;
	nFilter = [64]; lenFilter = [4]; lenPool = [8];
	nLSTM = [48, 48];
	'''
	
	# _____ Parameters
	IdTalkerAll = [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
	labelEmotion = ['anger', 'boredom', 'disgust','fear','happiness','sadness','neutral'] 
	TimeForced = 512		# Forced (either by padding or truncating) number of time frames for each speech file

	#startTime = time.time() 

	# _____ Load data & static feature extraction (feature = 1st & 2nd order statistics over time)
	print 'Loading data & feature extraction...'
	nClass = 7
	#X, Xtarget, Xinfo = emo.load_emodb_ftr_ext(modeFsel=modeFsel)
	X, Xtarget, Xinfo = emo.load_emodb_ftr_ext_new(modeFsel=modeFsel)
	emo.summary_dbinfo(Xinfo)
		
	nFile = (X.shape)[0]
	dimFeatIn = (X.shape)[2]
	dimTime = TimeForced
	if modeTD:
		dimFeatOut = dimFeatIn * 2
	else:
		dimFeatOut = dimFeatIn
	
	# _____ Talker-dependent normalization
	#	XnormStatic[:,dimFeatOut*2] is for SVM-based static approach only
	##Xnorm, XnormStatic = emo.feature_normalization_talker(X, Xinfo, modeTD, normEmotion='All')

	#_____ Assign talker id for hyperparameter search in LOSO-based cross-validation
	z=Xinfo['IdTalker']
	groups = np.zeros(nFile,int)
	for h in range(0,len(IdTalkerAll)):
		groups[np.where(z==IdTalkerAll[h])] = h

	logo = LeaveOneGroupOut()
	nlogo = logo.get_n_splits(X, Xtarget, groups)
	z=logo.split(X, Xtarget, groups)
	fidxSplit = [None]*nlogo
	for h in range(0,len(IdTalkerAll)):
		fidxSplit[h] = z.next()

	# _____ Model training and evaluation		
	if strModel=='svm':
		if modeSpkrdep=='sdall-sdn1' or modeSpkrdep=='sdn1':
			print 'Speaker-dependency mode selection is not valid for SVM.'
			raise SystemExit

		# _____ Train & Test for individual LOGO (LOSO)
		rateTrain = np.zeros(nlogo); rateTest = np.zeros(nlogo)
		yPredictAcc = []; yTrueAcc = [];
		for h in range(0, nlogo):
			train_index, test_index = fidxSplit[h]										
			xTrain = (XnormStatic[train_index], Xtarget[train_index])
			xTest = (XnormStatic[test_index], Xtarget[test_index])			
			#rateTrain[h], rateTest[h] = emo.run_svm(xTrain, xTest, randSeed)
			rateTrain[h], rateTest[h], yTrain, yTest = emo.run_svm(xTrain, xTest, 128.0, randSeed)
			print('%3d	%.3f	%.3f' % (h, rateTrain[h], rateTest[h]))
			#
			yPredict = np.array(yTest)
			yTrue = Xtarget[test_index];	
			yPredictAcc = np.concatenate((yPredictAcc, yPredict))
			yTrueAcc = np.concatenate((yTrueAcc, yTrue))
			cnfMatrix = metrics.confusion_matrix(yTrue, yPredict)			

		print('Avg Std:	%.3f	%.3f	%.3f	%.3f' %(np.mean(rateTrain), np.std(rateTrain), np.mean(rateTest), np.std(rateTest)))
		#
		cnfMatrixAcc = metrics.confusion_matrix(yTrueAcc, yPredictAcc)
		emo.plot_confusion_matrix(cnfMatrixAcc, classes=labelEmotion, title='Confusion matrix')
	
	else:	

		# _____ Target value for the classifier outputs
		XtargetNetout = np_utils.to_categorical(Xtarget, nClass)						
		
		# _____ Filename
		mdlStr = strModel+'_tdmode'+str(modeTD)+'_modeFsel'+str(modeFsel)+'_D'+str(dimFeatOut)+'T'+str(dimTime)
		if strModel == 'svm':
			mdlStr += '_rand' + str(randSeed)
		else:
			if nLayerCNN:
				mdlStr += '-CNN'
			for k in range(0,nLayerCNN):
				mdlStr += '-'+str(nFilter[k])+'x'+str(lenFilter[k])+'-mp'+str(lenPool[k])
			if nLayerLSTM:
				mdlStr += '-LSTM'
			for k in range(0,nLayerLSTM):	
				mdlStr += '-'+str(nLSTM[k])	
			mdlStr += '_fdrp' + str(fDropout) + '_bs' + str(batchSize) + '_rand' + str(randSeed)
			mdlStr += '_spkr' + modeSpkrdep
		
		# _____ Train & Test for individual LOGO (LOSO)

		#for h in range(0, nlogo):
		for h in range(0, 5):

			# normalization
			XnormSm, XnormStatic = emo.feature_normalization(X, Xinfo, idxLOSO=h, modeSpkrdep=modeSpkrdep)
			#id = np.unravel_index(XnormSm.argmax(), XnormSm.shape); plt.clf(); plt.plot(XnormSm[id[0],:,16])    # iFormantGain can be problematic

			if modeFsel==0 or modeFsel==2:
				dimUVChange = [14, 15, 16]
			else:
				print 'TD feature in main should be updated.'
				raise SystemExit

			if modeTD:
				XnormSm = emo.get_tdfeature(XnormSm, Xinfo)
				if modeFsel==0:
					dimUVChange.extend([34, 35, 36])
				elif modeFsel==2:   # non-TD dim = 24
					dimUVChange.extend([38, 39, 40])

			XnormSm = emo.smooth(XnormSm, a=0.9)    # smoothing
			XnormSm = emo.set_uvftrval(XnormSm, Xinfo, dimUVChange)
			XnormSmTN = emo.reshape_feature_fixedtime(XnormSm, np.array(Xinfo.nFrame), TimeForced)  # zero-padding & truncating

			train_index, test_index = fidxSplit[h]
			xTrain = XnormSmTN[train_index]; xTrainTarget = XtargetNetout[train_index]
			xTest = XnormSmTN[test_index]; xTestTarget = XtargetNetout[test_index]
			print xTrain.shape
			np.random.seed(randSeed)
			
			if strModel=='lstm' or strModel=='conv1gf':	#LSTM only OR Conv1gf
				model = emo.build_conv1Dgf_lstm(dimTime=dimTime, dimFeat=dimFeatOut, nFilter=nFilter, lenFilter=lenFilter, lenPool=lenPool, \
									nLSTM=nLSTM, fDropout=fDropout, nClass=nClass)
			else:
				print("Backend: ", keras.backend._backend, ", Image Ordering: ", keras.backend._image_data_format)
				# Reshape inputs, which is necessary to use Convolution2D()
				if keras.backend._image_data_format == 'channels_last':
					a = xTrain.shape; xTrain = np.reshape(xTrain, (a[0], a[1], a[2], 1))
					a = xTest.shape; xTest = np.reshape(xTest, (a[0], a[1], a[2], 1))
				elif keras.backend._image_data_format == 'channels_first':
					a=xTrain.shape; xTrain=np.reshape(xTrain,(a[0],1,a[1],a[2]))
					a=xTest.shape; xTest=np.reshape(xTest,(a[0],1,a[1],a[2]))

				if strModel=='conv1lf-conv1gf':	# Conv1Lf-Conv1Gf
					model = emo.build_conv1Dlf_conv1Dgf_lstm(dimTime=dimTime, dimFeat=dimFeatOut, nFilter=nFilter, lenFilter=lenFilter, lenPool=lenPool, \
										nLSTM=nLSTM, fDropout=fDropout, nClass=nClass)
				elif strModel=='conv2-conv1gf':	# Conv2-Conv1Gf
					model = emo.build_conv2D_conv1Dgf_lstm(dimTime=dimTime, dimFeat=dimFeatOut, nFilter=nFilter, lenFilter=lenFilter, lenPool=lenPool, \
										nLSTM=nLSTM, fDropout=fDropout, nClass=nClass)				
				elif strModel=='conv2':	# Conv2
					model = emo.build_conv2D(dimTime=dimTime, dimFeat=dimFeatOut, nFilter=nFilter, lenFilter=lenFilter, lenPool=lenPool, \
										fDropout=fDropout, nClass=nClass)
				
				
			fNameBase = 'h' + str(h) + '-' + mdlStr
			#fNameWeight = fNameBase + '_weight.best.hdf5'
			fNameLog = keras.callbacks.CSVLogger(dirOut + '/' + fNameBase + '-training.log')
			#fNameLog = keras.callbacks.CSVLogger(keras.backend._backend + '-' + keras.backend._image_data_format + '.log')
			callbacks_list = [fNameLog]
			if flagSaveModel:
				fNameModel = dirOut + '/' + 'model_' + fNameBase + '.json'
				#fNameWeight = dirOut + '/' + 'weight_' + fNameBase + '--{epoch:02d}-{val_acc:.2f}.hdf5'
				fNameWeight = dirOut + '/' + 'weight_' + fNameBase + '.hdf5'
				# Save weight only if validation accuracy is the best so far
				checkpoint = ModelCheckpoint(fNameWeight, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
				callbacks_list = [fNameLog, checkpoint]
			#earlystop = EarlyStopping(monitor='val_acc',patience=80,verbose=1)
			#callbacks_list = [fNameLog, checkpoint, earlystop]	
			
			
			# Train - use test data as validation data: this is valid as CV-validation is being on already in talker-level
			#startTime = time.time()

			history = model.fit(xTrain, xTrainTarget, validation_data=(xTest, xTestTarget), \
							epochs=epochLen, batch_size=batchSize, shuffle=True, verbose=1, callbacks=callbacks_list)
	
			#endTime = time.time()
			#print endTime-startTime
			
			# Train with K-fold CV on training data only
			#history = model.fit(xTrain, xTrainTarget, validation_split=0.2, nb_epoch=250, batch_size=batchSize, shuffle=True, verbose=1, callbacks=callbacks_list)		
			
			# Save model
			if flagSaveModel:			
				model_json = model.to_json()	# serialize model to JSON
				with open(fNameModel, "w") as json_file:
					json_file.write(model_json)		
				#model.save_weights(fNameWeight)	# serialize weights to HDF5
	
	#endTime = time.time()
	#print endTime-startTime
		
		
if __name__ == "__main__":
	main()
	
	
	
