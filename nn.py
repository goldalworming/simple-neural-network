from numpy import *
from numpy.random import *

p = array([[3,3,3],
	[3,2,2],
	[3,2,1],
	[3,1,1],
	[2,3,2],
	[2,2,2],
	[2,2,1],
	[2,1,1],
	[1,3,2],
	[1,2,1],
	[1,1,2]])
	
t = array([[1],
	[1],
	[1],
	[0],
	[1],
	[1],
	[1],
	[0],
	[1],
	[0],
	[1]])

jumpola = len(p)
dimpola = len(p[0])
JOneuron = 1 #len(t[0])

JHneuron = 5
LR = 0.1
Epoch = 1
maxMSE = 10**-5

############################
#import random
import math

w1 = uniform(low=-1,high=1,size=(dimpola, JHneuron))
w2 = uniform(low=-1,high=1,size=(JHneuron, JOneuron))
#w2 = rand(JHneuron, JOneuron)

print 'w1',w1
print 'w2',w2
	
MSEepoch = maxMSE +1
MSE = array([])
ee = 1

while (ee<=Epoch) and (MSEepoch > maxMSE):
	MSEepoch = 0
	for pp in range(0,jumpola):
		CP = p[pp,:]
		CT = t[pp,:]
		
		'''
		perhitungan maju
		'''			
		
		A1 = array([])
		for ii in range(0,JHneuron):
			v = dot(CP,w1[:,ii])
			A1 = append(A1, 1.0/(1+math.e**-v))
#		print 'A1', A1
		
		A2 = array([])
		for jj in range(0,JOneuron):
			v = dot(A1,w2[:,jj])
			A2 = append(A2,1.0/(1+math.e**-v))
		print 'A2', A2
		Error = CT-A2
#		print 'err', Error

		
		for kk in Error:
			MSEepoch = MSEepoch+kk**2
		print 'MSEepoch', MSEepoch	
		'''
		perhitungan mundur
		'''	
		D2 = array([])
		for kk in range(0,JOneuron):
			d = A2[kk]* (1-A2[kk])*Error[kk]
			D2 = append(D2,d)
		print 'D2', D2
		
		dW2 = array([])
		for jj in range(0, JHneuron):
			for kk in range(0,JOneuron):
				delta2 = LR*D2[kk]*A1[jj]
			dW2 = append(dW2,delta2)
#		print 'dW2', dW2
		
		D1 = array([])
		for jj in range(0,JHneuron):
			d = dot(A1,(1-A1).transpose())*D2*(w2[jj,:]).transpose()
			D1 = append(D1,d)		
		print '---------------D1',D1

		dW1 = array([])
		for ii in range(0, dimpola):
			delta1 = array([])
			for jj in range(0,JHneuron):
#				print 'LR', LR
#				print 'D1[jj]', D1[jj]
#				print 'CP[ii]', CP[ii]
				delta1 = append(delta1,LR*D1[jj]*CP[ii])
#			print 'delta1',delta1
			dW1 = append(dW1,delta1)
		w1 = w1+dW1.reshape(w1.shape)
		w2 = w2+dW2.reshape(w2.shape)
	MSE = append(MSE, MSEepoch/jumpola)
#	print 'MSE', MSE
	
	ee += 1			

print 'w1',w1
print 'w2',w2
