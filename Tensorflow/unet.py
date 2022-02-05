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
InputDimension=[320,240,3]
GtDimensions=[320,240,1]
LearningRate=1e-4
NumIteration=1e6


TrainInData=np.load('images.npy')
TrainOutData=np.load('masks.npy')
TrainOutData=TrainOutData/255.0

print("Image Preload Done")
#Number of images is 21600
NumTrainImages=TrainInData.shape[0]
print(NumTrainImages)

Train_x = tf.data.Dataset.from_tensor_slices(TrainInData[:2000,:,:,:])
Train_y = tf.data.Dataset.from_tensor_slices(TrainOutData[:2000,:,:])
Test_x = tf.data.Dataset.from_tensor_slices(TrainInData[2000:3000,:,:,:])
Test_y = tf.data.Dataset.from_tensor_slices(TrainOutData[2000:3000,:,:])
train_dataset = tf.data.Dataset.zip((Train_x,Train_y)).shuffle(500).repeat().batch(BatchNum)
test_dataset = tf.data.Dataset.zip((Test_x,Test_y)).shuffle(500).repeat().batch(BatchNum)

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
testing_init_op = iterator.make_initializer(train_dataset)







CurrentInput =tf.cast(next_element[0],tf.float32)
InputGT=tf.expand_dims(tf.cast(next_element[1],tf.float32),-1)

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
	Sess.run(training_init_op)  
	Step = 1
	while Step < NumIteration:
		#execute teh session
		_,L,OutputImages,ne = Sess.run([Optimizer, SoftL1Loss ,Enhanced,next_element])
		#print loss and accuracy at every 10th iteration
		if (Step%10)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Loss:" + str(L))
		#save samples
		if (Step%500)==0:
			for i in range(3):
				temp = (OutputImages[i,:,:])
				cv2.imwrite('samples/gt_'+str(Step).zfill(5)+'_'+str(i)+'.png',(ne[1][i,:,:]*255.0))
				cv2.imwrite('samples/in_'+str(Step).zfill(5)+'_'+str(i)+'.png',(ne[0][i,:,:,:]))
				cv2.imwrite('samples/out_'+str(Step).zfill(5)+'_'+str(i)+'.png',(temp*255.0))
		if (Step%5000)==0:
			print('Saving model...')
			print(Saver.save(Sess, "./checkpoint/"))
		Step+=1
