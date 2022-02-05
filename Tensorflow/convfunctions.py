import tensorflow as tf
import numpy as np


def LeakyReLU(Input):
	#leaky ReLU
	alpha=0.01
	Output =tf.maximum(alpha*Input,Input)
	return Output

def ConvBlock(Input,NumConv,KernelSize,LayerNum):
	#first convolution to scale the kernels
	with tf.variable_scope('conv'+str(LayerNum)):
		W = tf.get_variable('W',KernelSize)

		Input= tf.nn.conv2d(Input,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
	
		Bias  = tf.get_variable('B',[KernelSize[3]])
		Input= tf.add(Input, Bias)
		Input= LeakyReLU(Input)
		LayerNum+=1

	KernelSize[2]=KernelSize[3]
	for i in range(NumConv-1):
		with tf.variable_scope('conv'+str(LayerNum)):
			W = tf.get_variable('W',KernelSize)	
			Input= tf.nn.conv2d(Input,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
		
			Bias  = tf.get_variable('B',[KernelSize[3]])
			Input= tf.add(Input, Bias)

			Input= LeakyReLU(Input)
			LayerNum+=1
	return Input,LayerNum
