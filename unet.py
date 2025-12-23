import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob as glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from keras.metrics import MeanIoU
from tensorflow.keras.layers import input,Conv2D,MaxPooling2D,Conv2DTranspose,concatenate 
from tensorflow.keras.models import Model 

found_mask= False

for folder_path in folder_paths: 
    for file_path in sorted(glob(folder_path+"/*")):
        img = cv2.Imread(file_path)
        img = cv2.resize(img,cv2.COLOR_RGB2GRAY)
        img = img/255.0

        if "mask" in file_path:
            if found_mask:
                
                masks[-1]+= img
                masks[-1] = np.where(masks[-1]>0.5,1.0,0.0)
            else:
                masks.append(img)
                found_mask = True
        else:
            images.append(img)
            found_mask=False


            
X=np.array(images)  
y=np.array(masks) 

X=np.expand_dims(X,-1)
y=np.expand_dims(Y,-1)

print(f"X shape:{X.shape} |y shape:{y.shape}")

X_train, X_val,y_train,y_val=train_test_split(X,y,test_size=0.1)

size=128 

input_layer=input(shape=(size,size,1))

conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)


conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)


conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)


conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)


bottleneck = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
bottleneck = Conv2D(1024, (3, 3), activation="relu", padding="same")(bottleneck)

upconv1 = Conv2DTranspose(512, (2, 2), strides=2, padding="same")(bottleneck)
concat1 = concatenate([upconv1, conv4])
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(concat1)
conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

# # Second decoder block
upconv2 = Conv2DTranspose(256, (2, 2), strides=2, padding="same")(conv5)
concat2 = concatenate([upconv2, conv3])
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(concat2)
conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

# # Third decoder block
upconv3 = Conv2DTranspose(128, (2, 2), strides=2, padding="same")(conv6)
concat3 = concatenate([upconv3, conv2])
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat3)
conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

# # Fourth decoder block
upconv4 = Conv2DTranspose(64, (2, 2), strides=2, padding="same")(conv7)
concat4 = concatenate([upconv4, conv1])
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(concat4)
conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

output_layer=Conv2D(1,(1,1),activation="sigmoid",padding="same")(conv8)
model=Model(inputs=input_layer,outputs=output_layer)

model.compile(loss="binary_crossentropy",opyimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=40,validation_data=(X_val,y_val),verbose=1)

pred=model.predict(X_val,verbose=1)
pred=(pred>0.5).astype(int)
y_true=y_val.astype(int)

iou=jaccard_score(pred.flatten(),y_true())
print(f"IoU(Jaccard Score):{iou:.4f}")

mean_iou=MeanIoU(num_classes=2)
mean_iou.update_state(pred,y_true)
print("mean IoU=",mean_iou.result().numpy())


import matplotlib.pyplot as plt
import numpy as np

i = 6

# Plot original image
plt.subplot(1, 3, 1)
plt.imshow(X_val[i], cmap="gray")

# Plot ground truth mask
plt.subplot(1, 3, 2)
plt.imshow(y_val[i], cmap="gray")

# Plot model prediction
plt.subplot(1, 3, 3)
# Expand dims to simulate batch size of 1 for prediction
pred = model.predict(np.expand_dims(X_val[i], axis=0), verbose=1)[0]
# Binarize the output (thresholding at 0.5)
pred = (pred > 0.5) 
plt.imshow(pred, cmap="gray")

plt.show()

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# conv1 block
conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(input_layer)
conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)