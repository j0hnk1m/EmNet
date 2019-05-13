'''
	EmNet data analysis

'''

import re
import argparse
import sys
import csv
import platform
import pandas as pd
import numpy as np

import matplotlib

if platform.system() == 'Darwin':
	import appnope
	appnope.nope()
	matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def readEmotxt(inFile):
	'''
		Read emo_summary file (output of emo_summary2.sh) and return dataframe
	'''
	# Write header
	hdr = 'model,tdmode,modeFsel,Dim,nFrame,nFilter1,LFilter1,mp1,nFilter2,LFilter2,mp2,LSTM1,LSTM2,fDrop,batchS,spkr,rand'
	h1 = 'h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,Avg'
	hdr += ',' + h1 + 'Tst,' + h1 + 'Tr,' + h1 + 'Itr\n'

	with open(inFile) as file:
		lines = file.readlines()

	nExp = len(lines) / 5
	DX = [];
	n = 0;
	cnt = 0
	while n < len(lines):
		arch = lines[n].strip()

		# ----- Parse hyperparameters from the log file name
		k = arch.find('_tdmode')
		dx = {'fName': arch, 'model': arch[0:k], 'tdmode': int(arch[k + 7])}

		k = arch.find('_modeFsel') + 9
		dx['modeFsel'] = int(arch[k])

		base = arch[k + 1:]
		j1 = base.find('_D') + 2
		j2 = base.find('T')
		j3 = base.find('-CNN')
		dx['Dim'] = int(base[j1:j2])
		dx['nFrame'] = int(base[j2 + 1:j3])

		params = re.match(r'CNN-(.*)x(.*)-mp(.*)-(.*)x(.*)-mp(.*)-LSTM', arch[arch.find('CNN'):])
		p = [int(params.group(i)) for i in range(1, 7)]
		dx.update({'nFilter1': p[0], 'LFilter1': p[1], 'mp1': p[2], 'nFilter2': p[3], 'LFilter2': p[4], 'mp2': p[5]})

		a1 = arch.find('LSTM-') + 5
		a2 = arch.find('_fdrp')
		a_ = arch[a1:a2].find('-')  # for 2-layer LSTM
		if a_ > -1:
			a_ += a1
			p = [int(arch[a1:a_]), int(arch[a_ + 1:a2])]
		else:
			p = [int(arch[a1:a2]), 0]
		dx.update({'LSTM1': p[0], 'LSTM2': p[1]})

		params = re.match(r'fdrp(.*)_bs(.*)_rand(.*)_spkr(.*)-training.log', arch[a2 + 1:])
		if params == None:
			params = re.match(r'fdrp(.*)_bs(.*)_rand(.*)-training.log', arch[a2 + 1:])
			p = [str(params.group(i)) for i in range(1, 4)]
			p += ['sdall']
		else:
			p = [str(params.group(i)) for i in range(1, 5)]
		dx.update({'fDrop': float(p[0]), 'batchS': int(p[1]), 'rand': int(p[2]), 'spkr': p[3]})

		# ----- Data field
		itrN = lines[n + 1].strip().split(",")  # iterations
		rateTr = lines[n + 2].strip().split(",")  # result on training set
		rateTst = lines[n + 3].strip().split(",")  # result on validatino set
		dx.update({'AvgItr': float(itrN[-1]), 'AvgTr': float(rateTr[-1]), 'AvgTst': float(rateTst[-1])})

		for k in range(0, 10):
			dx.update({'rateTst' + str(k): float(rateTst[k + 1])})

		DX.append(dx.copy())

		cnt += 1
		n += 5  # there are 5 lines for each experimental condition, including a new line

	df = pd.DataFrame.from_dict(DX)

	msg = [(s, df[s].unique()) for s in list(df) if
	       ((df[s].nunique() > 1) and ('Avg' not in s) and ('rateTst' not in s))]
	print '### Independent variables:'
	for v in zip(msg):
		print v

	# Print the value of each condition that has only single value
	msg = [(s, df[s].unique()[0]) for s in list(df) if
	       ((df[s].nunique() == 1) and ('Avg' not in s) and ('rateTst' not in s))]
	print '### Controlled variables:'
	for v in zip(msg):
		print v

	return df


def readEmocsv(csvFile):
	dfOrg = pd.read_csv(csvFile)
	strCondition = ['modeFsel', 'tdmode', 'nFilter1', 'LFilter1', 'mp1', 'nFilter2', 'LFilter2', 'mp2', 'LSTM1',
	                'LSTM2',
	                'fDrop', 'batchS', 'spkr', 'rand']
	df = dfOrg[strCondition + ['AvgTst', 'AvgTr', 'AvgItr']]
	df = df.sort_values(strCondition)

	# Check independent variables
	# msg = [(s, df[s].nunique()) for s in list(df) if 'Avg' not in s]
	# msg = [(s, df[s].nunique()) for s in list(df) if ((df[s].nunique() > 1) and ('Avg' not in s))]
	msg = [(s, df[s].unique()) for s in list(df) if ((df[s].nunique() > 1) and ('Avg' not in s))]
	print '### Independent variables:'
	for v in zip(msg):
		print v

	# Print the value of each condition that has only single value
	msg = [(s, df[s].unique()[0]) for s in list(df) if ((df[s].nunique() == 1) and ('Avg' not in s))]
	print '### Controlled variables:'
	for v in zip(msg):
		print v

	return df, dfOrg

sns.set_style("whitegrid")
medianprops = dict(linewidth=2)

# _______________
#inFile = 'log5_pv02_lstm/result_rand1.txt'
nLSTM = [36, 48, 64, 96, 128]
x1 = [67.2, 70.7, 71.9, 72.8, 70.8]
x2 = [71.1, 69.1, 73.7, 73.4, 72.6]
ax1, = plt.plot(nLSTM, x1, 'o-', label = '1 Layers')
ax2, = plt.plot(nLSTM, x2, 'o-', label = '2 Layers')
plt.legend(handles=[ax1, ax2])
plt.xlabel('LSTM Size')
plt.ylabel('Recognition Rate [%]')
plt.title('LSTM performance')


# _______________ Compare pv
inFile = 'result.log5_spkr_pv01.txt'
df01 = readEmotxt(inFile)
inFile = 'result.log5_spkr_pv02.txt'
df = readEmotxt(inFile)
df02 = df[(df.tdmode == 0) & (df.LSTM1 == 48) & (df.LSTM2 == 0) & (df.nFilter2 == 96) & (df.spkr == 'sdall')].copy()
inFile = 'result.log5_spkr_pv03.txt'
df03 = readEmotxt(inFile)
inFile = 'result.log5_spkr_pv04.txt'
df = readEmotxt(inFile)
df04 = df[(df.tdmode == 0) & (df.LSTM1 == 48) & (df.LSTM2 == 0) & (df.nFilter2 == 96) & (df.spkr == 'sdall')].copy()

df01['pv']=0.1
df02['pv']=0.2
df03['pv']=0.3
df04['pv']=0.4
df = pd.concat([df01,df02,df03,df04])
ax = sns.boxplot(x='pv', y='AvgTst', data=df, linewidth=1, width=0.5, showmeans=True)
sns.swarmplot(x="pv", y="AvgTst", color='red', alpha=0.4, data=df )
plt.title('CNN-64x8x3-96x3x4-LSTM48')


# _______________ pv=0.2: best performance
inFile = 'result.log5_spkr_pv02.txt'
df = readEmotxt(inFile)
# All
ax = df.boxplot(column='AvgTst', by=['tdmode', 'nFilter2', 'LSTM1', 'LSTM2'], medianprops=medianprops, showmeans=True, fontsize=7, rot=45)

# Comparison of nFilter2
df2=df[(df.tdmode == 0) & (df.LSTM1 == 48) & (df.LSTM2 == 0)].copy()
plt.figure()
ax = sns.boxplot(x='nFilter2', y='AvgTst', data=df2, linewidth=1, width=0.5, showmeans=True)
sns.swarmplot(x="nFilter2", y="AvgTst", color='red', alpha=0.4, data=df2)
plt.title('CNN-64x8x3-nFilter2x3x4-LSTM48')

# Comparison of LSTM
df2=df[(df.tdmode == 0) & (df.nFilter2 == 96)].copy()
ax = df2.boxplot(column='AvgTst', by=['LSTM1', 'LSTM2'], medianprops=medianprops, showmeans=True, fontsize=7, rot=45)
plt.figure()
ax = sns.boxplot(x='LSTM1', y='AvgTst', hue='LSTM2', data=df2, linewidth=1, width=0.5, showmeans=True)
plt.title('CNN-64x8x3-96x3x4-LSTM1-LSTM2')


# _______________ SA/SI with pv=0.4
inFile = 'result.log5_spkr_pv04.txt'
df = readEmotxt(inFile)
# All
ax = df.boxplot(column='AvgTst', by=['tdmode', 'LSTM1', 'spkr'], medianprops=medianprops, showmeans=True, fontsize=7, rot=45)

#
df2 = df[(df.tdmode == 0) & (df.LSTM1 == 48) & (df.spkr == 'si')].copy()
plt.figure()
ax1, = plt.plot(df2['rand'], df2['AvgTst'], 'o', label='tst')
ax2, = plt.plot(df2['rand'], df2['AvgTr'], 'o', label='tr')
plt.xlabel('Random Seed')
plt.legend(handles=[ax1, ax2])
plt.legend((ax1,ax2),('Test','Training'))
plt.title('SI-mode performance with CNN-64x8x3-96x3x4-LSTM48')

