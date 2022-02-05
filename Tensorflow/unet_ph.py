import tensorflow as tf
import numpy as np
import os
import cv2
import random
import convfunctions as conv

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BatchNum=8
InputDimension=[240,320,3]
GtDimensions=[240,320,1]
LearningRate=1e-4
NumIteration=3006


TrainInData=np.load('images.npy')
TrainOutData=np.load('masks.npy')
TrainOutData=TrainOutData/255.0

print("Image Preload Done")
#Number of images is 21600
NumTrainImages=TrainInData.shape[0]
print(NumTrainImages)

InputDataPh = tf.placeholder(tf.float32, [BatchNum]+InputDimension)
ExpectedOutputPh = tf.placeholder(tf.float32, [BatchNum]+GtDimensions)








CurrentInput =InputDataPh
InputGT=ExpectedOutputPh

NumKernels=[InputDimension[2],8,16,32,64]
ImgSizes=[]
LayerLefts=[]
LayerNum=0
for n in range(0,len(NumKernels)-2):
   with tf.variable_scope('conv'+str(LayerNum)):
      LayerLeft,LayerNum = conv.ConvBlock(CurrentInput ,1,[3,3,NumKernels[n],NumKernels[n+1]],LayerNum)
      W = tf.get_variable('Wdown',[3, 3, NumKernels[n+1],NumKernels[n+1]])
      CurrentInput=tf.nn.conv2d(LayerLeft,W,strides=[1,2,2,1],padding='SAME')
      print(CurrentInput)
      LayerLefts.append(LayerLeft)
      ImgSizes.append([int(LayerLeft.get_shape()[1]),int(LayerLeft.get_shape()[2])])
      LayerNum +=1
CurrentInput,LayerNum = conv.ConvBlock(CurrentInput ,2,[3,3,NumKernels[-2],NumKernels[-1]],LayerNum)

print(CurrentInput)
for n in range(len(NumKernels)-1,1,-1):
	with tf.variable_scope('conv'+str(LayerNum)):
		W = tf.get_variable('W',[3, 3, NumKernels[n-1], NumKernels[n]])
		LayerRight=tf.nn.conv2d_transpose(CurrentInput, W,  [BatchNum, ImgSizes[n-2][0], ImgSizes[n-2][1], NumKernels[n-1]], [1, 2, 2 , 1], padding='SAME', name=None)
		print(LayerRight)
		Bias  = tf.get_variable('B',[NumKernels[n-1]])
		LayerRight=tf.add(LayerRight,Bias )
		LayerRight= conv.LeakyReLU(LayerRight)
		LayerNum +=1
		CurrentInput=tf.concat([LayerRight,LayerLefts[n-2] ],3)
		CurrentInput,LayerNum= conv.ConvBlock(CurrentInput ,2,[3,3,NumKernels[n],NumKernels[n-1]],LayerNum)

with tf.variable_scope('conv'+str(LayerNum)):
	W = tf.get_variable('W',[3, 3, NumKernels[1], GtDimensions[-1]])
	LayerOut= tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='SAME') #VALID, SAME
	Bias  = tf.get_variable('B',[GtDimensions[-1]])
	LayerOut= tf.add(LayerOut, Bias)
#no nonlinearity at the end
#LayerOut= LeakyReLU(LayerOut)
Enhanced=tf.nn.sigmoid(LayerOut)



# Define loss and optimizer
with tf.name_scope('loss'):
    # L1 loss
    AbsDif=tf.abs(tf.subtract(InputGT,Enhanced))

    #this part implements soft L1
    Comp = tf.constant(np.ones(AbsDif.shape), dtype = tf.float32)
    SmallerThanOne = tf.cast(tf.greater(Comp, AbsDif),tf.float32)
    LargerThanOne= tf.cast(tf.greater(AbsDif, Comp ),tf.float32)   
    ValuestoKeep=tf.subtract(AbsDif, tf.multiply(SmallerThanOne ,AbsDif))
    ValuestoSquare=tf.subtract(AbsDif, tf.multiply(LargerThanOne,AbsDif))
    SoftL1= tf.add(ValuestoKeep, tf.square(ValuestoSquare)) 
 
    #average loss
    SoftL1Loss = tf.reduce_mean( SoftL1)
    L1Loss = tf.reduce_mean( AbsDif)
    


with tf.name_scope('optimizer'):	
    #Use ADAM optimizer this is currently the best performing training algorithm in most cases
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(SoftL1Loss )

print("Computation graph building Done")

Init = tf.global_variables_initializer()

with tf.Session() as Sess:
	Sess.run(Init)
	Saver = tf.train.Saver()
	Step = 1
	while Step < NumIteration:
		SelectedIndices=np.random.choice(NumTrainImages, BatchNum, replace=False)
		batch_x=TrainInData[SelectedIndices,:,:,:]
		batch_y=np.reshape(TrainOutData[SelectedIndices,:,:], [BatchNum]+GtDimensions )
		


		#execute teh session
		_,L,OutputImages,InImages = Sess.run([Optimizer, SoftL1Loss ,Enhanced,InputDataPh],feed_dict={InputDataPh: batch_x , ExpectedOutputPh: batch_y})
		#print loss and accuracy at every 10th iteration
		if (Step%10)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Loss:" + str(L))
		#save samples
		if (Step%500)==0:
			for i in range(3):
				temp = (OutputImages[i,:,:])
				cv2.imwrite('l1_samples/gt_'+str(Step).zfill(5)+'_'+str(i)+'.png',(batch_y[i,:,:]*255.0))
				cv2.imwrite('l1_samples/in_'+str(Step).zfill(5)+'_'+str(i)+'.png',(batch_x[i,:,:,:]))
				cv2.imwrite('l1_samples/out_'+str(Step).zfill(5)+'_'+str(i)+'.png',(temp*255.0))
		if (Step%3000)==0:
			print('Saving model...')
			print(Saver.save(Sess, "./checkpoint/"))
		Step+=1
