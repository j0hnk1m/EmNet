'''
EmNet Evaluation by loading model & weight files
'''

# imports
import os
import sys
import platform
from EmNetLib import plot_confusion_matrix
sys.path.append("../data")

import matplotlib
if platform.system() == 'Darwin':
	import appnope
	appnope.nope()
	matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#import matplotlib.patheffects as PathEffects
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import itertools
import numpy.matlib

from sklearn import svm
from collections import Counter
from sklearn.metrics import classification_report
from sklearn import metrics

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, LSTM, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

import EmNetLib_work5 as emo

import keras
from keras.wrappers.scikit_learn import KerasClassifier, BaseWrapper
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold, LeaveOneGroupOut, GridSearchCV
import copy
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

from pylab import savefig

def main():
	
	modeFsel = 0
	modeTD = 0
	modeSpkrdep = 'sdall'
	TimeForced = 512
	
	dirModel='./log5_model/'
	nameModel = 'conv1lf-conv1gf_tdmode0_modeFsel0_D20T512-CNN-64x8-mp4-96x3-mp4-LSTM-48_fdrp0.5_bs64_rand1_spkrsdall.json'
	nameWgt = 'conv1lf-conv1gf_tdmode0_modeFsel0_D20T512-CNN-64x8-mp4-96x3-mp4-LSTM-48_fdrp0.5_bs64_rand1_spkrsdall.hdf5'

	# _____ Parameters
	IdTalkerAll = [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
	labelEmotion = ['anger', 'boredom', 'disgust','fear','happiness','sadness','neutral'] 
	
	# _____ Load data & static feature extraction (feature = 1st & 2nd order statistics over time)
	print 'Loading data & feature extraction...'
	nClass = 7
	X, Xtarget, Xinfo = emo.load_emodb_ftr_ext_new(modeFsel=modeFsel)		# X[nFile, MaxT, dimFeat]
	nFile = (X.shape)[0]
	dimFeatIn = (X.shape)[2]
	dimTime = TimeForced
	if modeTD:
		dimFeatOut = dimFeatIn * 2
	else:
		dimFeatOut = dimFeatIn

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

	# _____ Target value for the classifier outputs
	XtargetNetout = np_utils.to_categorical(Xtarget, nClass)						
				
	# _____ Test for individual LOGO (LOSO)
	score = np.zeros((nlogo,2));
	uar = np.zeros((nlogo,1));
	yPredictAcc = []; yTrueAcc = [];
	cntClassAcc = np.zeros(7, int)
	cntClassCorrectAcc = np.zeros(7, int)
			
	for h in range(0, nlogo):
		XnormSm, XnormStatic = emo.feature_normalization(X, Xinfo, idxLOSO=h, modeSpkrdep=modeSpkrdep)
		if modeFsel == 0 or modeFsel == 2:
			dimUVChange = [14, 15, 16]
		else:
			print 'TD feature in main should be updated.'
			raise SystemExit

		if modeTD:
			XnormSm = emo.get_tdfeature(XnormSm, Xinfo)
			if modeFsel == 0:
				dimUVChange.extend([34, 35, 36])
			elif modeFsel == 2:  # non-TD dim = 24
				dimUVChange.extend([38, 39, 40])

		XnormSm = emo.smooth(XnormSm, a=0.9)    # smoothing
		XnormSm = emo.set_uvftrval(XnormSm, Xinfo, dimUVChange)
		XnormSmTN = emo.reshape_feature_fixedtime(XnormSm, np.array(Xinfo.nFrame), TimeForced)  # zero-padding & truncating


		train_index, test_index = fidxSplit[h]
		xTrain = XnormSmTN[train_index]; xTrainTarget = XtargetNetout[train_index]
		xTest = XnormSmTN[test_index]; xTestTarget = XtargetNetout[test_index]		
		# Reshape inputs, which is necessary to use Convolution2D() 
		a=xTrain.shape; xTrain=np.reshape(xTrain,(a[0],a[1],a[2],1))
		#a=xTest.shape; xTest=np.reshape(xTest,(a[0],1,a[1],a[2]))
		a=xTest.shape; xTest=np.reshape(xTest,(a[0],a[1],a[2],1))

		# Load model
		fnameModel = dirModel + 'model_h'+str(h)+'-'+ nameModel
		fnameWgt = dirModel + 'weight_h'+str(h)+'-'+ nameWgt
		fileModel = open(fnameModel, 'r')
		modelLoadedJson = fileModel.read()
		fileModel.close()
		model = model_from_json(modelLoadedJson)		
		# Load weight
		model.load_weights(fnameWgt)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])	
		# Evaluation
		scoreTrain = model.evaluate(xTrain, xTrainTarget, verbose=0)
		score[h] = model.evaluate(xTest, xTestTarget, verbose=0)
		print "%s: Train: %.2f%%, Test: %.2f%%" % (model.metrics_names[1], scoreTrain[1]*100, score[h,1]*100)


		#
		yRaw = model.predict(xTest, verbose=0)
		yPredict = np.argmax(yRaw,axis=1)
		yTrue = Xtarget[test_index];	
		yPredictAcc = np.concatenate((yPredictAcc, yPredict))
		yTrueAcc = np.concatenate((yTrueAcc, yTrue))
		cnfMatrix = metrics.confusion_matrix(yTrue, yPredict)
		#plot_confusion_matrix(cnfMatrix, classes=labelEmotion, title='Confusion matrix')
		#
		cntClass = np.zeros(7, int)
		cntClassCorrect = np.zeros(7, int)
		for n in range(0, len(yTrue)):
			for k in range(0, 7):
				if yTrue[n] == k:
					cntClass[k] += 1
					cntClassAcc[k] += 1
					if yTrue[n] == yPredict[n]:
						cntClassCorrect[k] += 1
						cntClassCorrectAcc[k] += 1

		idx = np.nonzero(cntClass)
		uar[h] = np.average(np.divide(cntClassCorrect[idx].astype(float), cntClass[idx].astype(float)))			
		print "[h=%d] %s: %.2f%%  %.2f%%" % (h, model.metrics_names[1], score[h,1]*100, uar[h]*100.)		

	print "Average: %.2f%%  (UAR = %.2f%%)" %(100*np.mean(score[:,1],axis=0), 100*np.mean(uar))
	
	cnfMatrixAcc = metrics.confusion_matrix(yTrueAcc, yPredictAcc)
	plot_confusion_matrix(cnfMatrixAcc, classes=labelEmotion, title='Confusion matrix')
	# this can be different from mean(score), as it calculates the mean of accumulated samples
	scoreAcc = 100.*np.float(sum(int(a == b) for a, b in zip(yPredictAcc, yTrueAcc))) / np.float(len(yPredictAcc))
	print "Accumulated score across all LOSO: %.2f%% " %(scoreAcc)
	
	print cntClassCorrectAcc, cntClassAcc
	print np.average(np.divide(cntClassCorrectAcc.astype(float), cntClassAcc.astype(float)))	


############## ANALYSIS - select a specific h and do the followings


# --------------- Plot weights
W = model.layers[0].get_weights()
W = W[0].reshape(8, 64).transpose()

nW = W.shape[0]
slopeFit = np.zeros(nW, float)
x = [0, 1, 2, 3, 4, 5]
x = range(0, W.shape[1])
for k in range(0, nW):
	pc = np.polyfit(x, W[k], 1)
	slopeFit[k] = pc[0]
sortedSlope = np.sort(slopeFit)
idxSlope = np.argsort(slopeFit)
newW = W[idxSlope]

plt.clf();
plt.plot(newW[0], '.-');
prevMin = np.min(newW[0]) - gap
for k in range(1, 64):
	cur0 = prevMin - np.max(newW[k])
	plt.plot(newW[k, :] + cur0, '.-')
	prevMin = cur0 + np.min(newW[k]) - gap

# -------------------------------------
#  Look into LSTM final output vector (input to Dense) to see how well different classes are clusterd
# using tsne
get_out_LSTM = K.function([model.layers[0].input, K.learning_phase()], [model.layers[6].output])
RS = 20150
cnt = 1

# For training data
# for p in [2, 4, 10, 20, 40, 80, 200]:
for p in [2, 5, 10, 30, 50, 80, 100, 200]:
	x = xTrain.copy();
	xidx = train_index;
	# OR
	x = xTest.copy();
	xidx = test_index;

	strTitle = 'p: ' + str(p)
	# strTitle = 'training (' + str(100 * scoreTrain[1])[:5] + '%)'

	# For test data
	# p=7; x = xTest.copy(); xidx = test_index; strTitle = strCond + ', test (' + str(100*score[h,1])[:5] + '%)'

	# run tSNE
	xLSTM = get_out_LSTM([x, 0])[0]
	# plt.figure(3); plt.clf(); plt.imshow(xLSTM, aspect='auto')
	xTsne = TSNE(random_state=RS, perplexity=p, verbose=1, n_iter=5000).fit_transform(xLSTM)
	df = pd.DataFrame(xTsne, columns=['x', 'y'])
	df['emotionID'] = Xinfo['IdEmo'][xidx].tolist()
	df['emotion'] = Xinfo['StrEmo'][xidx].tolist()
	med = df.groupby('emotion')[['x', 'y']].median()

	# plot
	plt.figure(cnt);
	plt.clf();
	ax = plt.scatter(df.x, df.y, c=df.emotionID, alpha=0.8, cmap='viridis')
	for i in range(0, len(med)):
		txt = plt.text(med.iloc[i, :][0], med.iloc[i, :][1], str(med.index[i]), fontsize=14, alpha=0.7)
		txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
	plt.xlabel('tSNE-x');
	plt.ylabel('tSNE-y');
	plt.title(strTitle)

	#savefig('p' + str(p) + '.pdf')
	cnt += 1

	# another way to plot
	colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k']
	plt.figure(2)
	for e in range(0, len(labelEmotion)):
		idx = df.index[df.emotionID == e].tolist()
		plt.scatter(df.loc[idx, 'x'], df.loc[idx, 'y'], color=colors[e], alpha=0.5, label=labelEmotion[e])
	plt.legend(frameon=True, borderaxespad=0)
	plt.xlabel('tSNE-x');
	plt.ylabel('tSNE-y');
	plt.title(strTitle)

	# 3d
	xTsne = TSNE(n_components=3, random_state=RS, perplexity=p, verbose=1).fit_transform(xLSTM)
	df = pd.DataFrame(xTsne, columns=['x', 'y', 'z'])
	df['emotionID'] = Xinfo['IdEmo'][xidx].tolist()
	df['emotion'] = Xinfo['StrEmo'][xidx].tolist()
	med = df.groupby('emotion')[['x', 'y']].median()
	ret = plt.figure().gca(projection='3d')
	ret.scatter(df.x, df.y, df.z, c=df.emotionID, alpha=0.8, cmap='viridis')
	plt.show()



if __name__ == "__main__":
	main()
	
