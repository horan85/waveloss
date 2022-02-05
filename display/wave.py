import cv2
import numpy as np
import scipy.ndimage as morph

#Loading test images
Img1=cv2.imread("obj1.png",0)/255.0
Img2=cv2.imread("obj2.png",0)/255.0


Union= np.maximum(Img1,Img2)
Intersect= np.minimum(Img1,Img2)

print(Union.shape)
print(Intersect.shape)

print(np.amax(Union))
Wave=np.copy(Intersect)

#Parameters
SpatialProp=3 #increase in space (width height) during propagation, propagates in 3x3 neighbourhoods
ValueProp=0.01 #increase in values during propagation
NumSteps=int(np.amax(Union)/ValueProp)+1

#the weights for the loss are the same in this setup
SpatialWeights=np.ones(NumSteps)
ValueWeights=np.ones(NumSteps)

Loss=0.0
for ind in range(NumSteps):
        #spatial propagation
        SpatInc=morph.grey_dilation(Wave, size=(SpatialProp,SpatialProp))
        SpatInc=np.minimum(SpatInc,Union)
        
        SpatialLoss=np.sum(SpatInc-Wave)
        Wave=SpatInc
        #value propagation
        ValueInc=Wave+ValueProp
        ValueInc=np.minimum(ValueInc,Union)
        
        ValueLoss=np.sum(ValueInc-Wave)
        Wave=ValueInc
        #Loss
        Loss+=(ValueWeights)*ValueLoss+(SpatialWeights)*SpatialLoss
        cv2.imshow("Union",Wave )
        cv2.waitKey(0)
