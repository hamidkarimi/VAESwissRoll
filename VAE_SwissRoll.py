import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import random

# This functions returns as many as numSamples samples from a multivarirant guassian distribution
def getMutiVarGaussian (numSamples, mean = [10, -10] , stdev =  [[10, -5], [-5, 10]]):	
	return np.random.multivariate_normal(mean, stdev, numSamples)

# Generating 3D Swiss Roll data
def generate3DSwissRoll(numSamples, mean = [10, -10] , stdev =  [[10, -5], [-5, 10]]):
	gaussianSamples =  getMutiVarGaussian(numSamples, mean, stdev)
	SRSamples = np.zeros([numSamples, 3 ])
	for i in range(numSamples):
		SRSamples[i][0] = gaussianSamples[i][0] * math.cos(gaussianSamples[i][0]) # x
		SRSamples[i][1] = gaussianSamples[i][1]					  # y
		SRSamples[i][2] = gaussianSamples[i][0] * math.sin(gaussianSamples[i][0]) # z

	return SRSamples
def addNoise(Samples, Keepprob = 1.7, mean = 2, stddev = 4):
	#print Samples.shape
	j= 0
	noisySamples = np.zeros([Samples.shape[0], 3 ])
	for i in range(Samples.shape[0]):
		rnd = random.random()
		if rnd < Keepprob:
			j += 1
			noise = np.random.normal(mean,stddev)			
			noisySamples[i][0] = Samples[i][0] + noise
			noisySamples[i][1] = Samples[i][1] + noise
			noisySamples[i][2] = Samples[i][2] + noise

	print j
	return noisySamples		
	
# returning next batch
def NextBach(Samples, batchSize):
	sampleSize = int(Samples.shape[0])
	for i in range(int(sampleSize/batchSize)):
		batch = Samples[i*batchSize:(i+1)*batchSize ] 
		yield batch

#Plotting generated and original data
def plotSwissRoll(original, generated = None, epoch = 1, title = 'Varitional Auto-Encoder. Red: original Blue: generated. Epoch: ' ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	color = 'r'
	if generated != None :				
		ax.scatter(generated[:,0], generated[:,1], generated[:,2],  color='b', label="plot A")

	else:
		color = 'b'
	ax.scatter(original[:,0], original[:,1], original[:,2],  color=color)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.title(title + str(epoch))

#Plotting Varitional Free Enegery curve. percent is percentage of data that should be plotted
def plotPerformanceLog(log, percent = 0.1):
	step = int(log.shape[0] * percent)
	log = np.array(log)
	fig = plt.figure()
	ax = fig.gca()
	ax.plot(log[:,0][::step],log[:,2][::step],'r') 
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Varitional Free Enegery')

#Xavier initilaization for weights 
# From https://www.tensorflow.org/api_docs/python/contrib.layers/initializers#xavier_initializer

def weight_init(fan_in, fan_out, constant=1): 
	low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
	high = constant*np.sqrt(6.0/(fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#==================================== Varitional Autoncoder ============================
class VAE(object):
	def __init__(self, batchSize , zNum , encoderH1Size , encoderH2Size , decoderH1Size , decoderH2Size ,  learningRate    ):
		self.batchSize = batchSize # each batch is 100 consecutive points of (x,y,z) from generated swiss roll 
		self.zNum = zNum # number of z latent variables (  dimentionality of z distribution) 
		self.encoderH1Size = encoderH1Size # Number of hidden units in 1st layer of enconding multilayer perceptron 
		self.encoderH2Size = encoderH2Size # Number of hidden units in 2nd layer of enconding MLP 
		self.decoderH1Size = decoderH1Size # Number of hidden units in 1st layer of decoder MLP
		self.decoderH2Size = decoderH2Size # Number of hidden units in 2nd layer of decoder MLP  
		self.inputSize = 3 # each sample is a triple (x,y,z)
		self.learningRate = learningRate 
		self.input = tf.placeholder(tf.float32, [self.batchSize, self.inputSize]) # placeholder for input samples
		self.noisyInput = tf.placeholder(tf.float32, [self.batchSize, self.inputSize]) # placeholder for input samples
		self.initWeights()
		self.network()
		self.optimization()

		init = tf.initialize_all_variables()
		self.sess = tf.InteractiveSession()
		self.sess.run(init)


	def initWeights(self):
		self.Weights = dict()
		# Weights of encoder
		self.Weights['encoder'] = {\
			'wh1': tf.Variable(weight_init(self.inputSize, self.encoderH1Size)),\
			'wh2': tf.Variable(weight_init(self.encoderH1Size, self.encoderH2Size)), \
			'wmean': tf.Variable(weight_init(self.encoderH2Size, self.zNum)),\
			'wstddev': tf.Variable(weight_init(self.encoderH2Size, self.zNum)), \
			'b1': tf.Variable(tf.zeros([self.encoderH2Size], dtype=tf.float32)),\
			'b2': tf.Variable(tf.zeros([self.encoderH2Size], dtype=tf.float32)),\
			'bmean': tf.Variable(tf.zeros([self.zNum], dtype=tf.float32)),\
			'bstddev': tf.Variable(tf.zeros([self.zNum], dtype=tf.float32))}

		# Weights of decoder
		self.Weights['decoder'] = {\
			'wh1': tf.Variable(weight_init(self.zNum, self.decoderH1Size)),\
			'wh2': tf.Variable(weight_init(self.decoderH1Size, self.decoderH2Size)),\
			'wx': tf.Variable(weight_init(self.decoderH2Size, self.inputSize)),\
			'b1': tf.Variable(tf.zeros([self.decoderH1Size], dtype=tf.float32)),\
			'b2': tf.Variable(tf.zeros([self.decoderH2Size], dtype=tf.float32)),\
			'bx': tf.Variable(tf.zeros([self.inputSize], dtype=tf.float32))}
	
	# Creates the network by calling the encoder and decoder functions as well as computing p(z)
	def network(self):
		self.zMean, self.zStddev = self.encoder() # Acquiring mean and stddev from encoder MLP
		epsilon = tf.random_normal((self.batchSize, self.zNum), 0, 1, dtype=tf.float32)  # epsilon is coming from a normal (0,1) distribution
		self.z = tf.add(self.zMean,tf.mul(tf.sqrt(tf.exp(self.zStddev)), epsilon)) # computing z from epsilon
		self.generatedX = self.decoder() # Decoder is called to generate x samples from p(x|z)


	# encodes x to mean and stddev of latent variables
	def encoder(self):
		l1 = tf.nn.tanh(tf.add(tf.matmul(self.noisyInput, self.Weights['encoder']['wh1']), self.Weights['encoder']['b1']))  # layer 1 tanh(X*Wh1+b2)
		l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.Weights['encoder']['wh2']), self.Weights['encoder']['b2']))  # layer 1 tanh(l1*Wh2+b2)
		mean = tf.add(tf.matmul(l2,self.Weights['encoder']['wmean']),self.Weights['encoder']['bmean']) # output layer yields mean = l2*Wmean+bmean
		stddev = tf.add(tf.matmul(l2,self.Weights['encoder']['wstddev']),self.Weights['encoder']['bstddev']) # output layer yields stddev = l2*Wmean+bmean
		return mean, stddev


	# constructs x from z
	def decoder(self):
		l1 = tf.nn.tanh(tf.add(tf.matmul(self.z, self.Weights['decoder']['wh1']), self.Weights['decoder']['b1'])) # layer 1 tanh(z*Wh1+b1)
		l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.Weights['decoder']['wh2']), self.Weights['decoder']['b2'])) # layer 1 tanh(l1*Wh2+b2)
		x = tf.add(tf.matmul(l2, self.Weights['decoder']['wx']),self.Weights['decoder']['bx']) # last layer is constructed x. Unlike binary output, here we take a linear activation for final value
		return x


	#defines latent and reconstruction losses and sets the optimizer
	# I followedd Auto-Encoding Variational paper by Bayes D.Kingma & M.Welling (page 5). 
	def optimization(self):
		
		self.reconstructionLoss = reconstr_loss = 0.5 * tf.reduce_sum( tf.square(self.input  - self.generatedX )) # sum of square root of input from reconstructed 
		self.KL = -0.5 * tf.reduce_sum(1 +  self.zStddev - tf.square(self.zMean) - tf.exp(self.zStddev), 1)  # KL divergence 
		self.totalcost = tf.reduce_mean(self.KL + self.reconstructionLoss)   # total cost
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.totalcost) # optimization 

	# running the session to extract results
	def runVAE(self, X, Xnoisy ):
		optimizer, reconstructionLoss, KL,totalcost, generatedX =\
			self.sess.run((self.optimizer, self.reconstructionLoss,  self.KL, self.totalcost, self.generatedX),feed_dict={self.input: X, self.noisyInput:Xnoisy})
        	return  optimizer, reconstructionLoss, KL,totalcost, generatedX

#======================================== Training =============================================

def train(InputSamples, noisySamples, numEpoch, batchSize = 100, zNum = 30, encoderH1Size = 100, encoderH2Size = 100, decoderH1Size = 100, decoderH2Size = 100,  learningRate = 0.0005 ):
	vae = VAE( batchSize , zNum , encoderH1Size , encoderH2Size , decoderH1Size , decoderH2Size ,  learningRate )	
	performanceLog = [] # Varitional Free Enegery
	i = 0
	reportInterval =100# int((InputSamples.shape[0]/batchSize) * 0.1) # 10% of iterations of each epoch are reported and logged

	for epoch in range(numEpoch):	
		generatedXs = []
		for idx, (batch, noisyBatch) in enumerate(zip(NextBach(InputSamples, batchSize), NextBach(noisySamples, batchSize) )):
			optimizer, reconstructionLoss, KL,totalcost, generatedX = vae.runVAE(batch,noisyBatch )			
			i = i + 1
			if epoch % reportInterval == 0:
				print "Epoch:",  (epoch+1),  "Varitional Free Enegery=", totalcost, "Reconstruction Loss=", reconstructionLoss,  "KL =", np.mean(KL)
				performanceLog.append((i,epoch,totalcost,reconstructionLoss,np.mean(KL))) # gathering Varitional Free Enegery i.e., total cost

			generatedXs.append(generatedX) # harvesting generated samples
		if epoch % 200 == 0 or (epoch <= 100 and epoch % 50 ==0) or (epoch <= 12 and epoch % 4 ==0):
				generatedXs = np.array(generatedXs)
				generatedXs = np.reshape(generatedXs, (InputSamples.shape[0],3))
				plotSwissRoll(original = generatedXs, generated = None, epoch = epoch+1, title ="Varitional Auto-Encoder. Generated plot. Epoch: ")
				plotSwissRoll(InputSamples, generatedXs, epoch+1)
	return performanceLog, generatedXs


def main():
	InputSamples = generate3DSwissRoll(numSamples = 10000)
	noisySamples = addNoise(InputSamples)
	performanceLog , generatedXs = train(InputSamples,noisySamples, numEpoch = 2001) # 1001 epoch
	performanceLog = np.array(performanceLog)
	with open("VAEResults/logVAE",'w') as  outputfile:
		for l in performanceLog:
			outputfile.write(str(l[0])+','+str(l[1])+','+str(l[2])+','+str(l[3])+','+str(l[4])+'\n') # Recording the performance metrics into file	
	plotPerformanceLog(performanceLog)
	plt.show()

main()




			
			

		
		




		
		
		
		
		 
		 



		
		
