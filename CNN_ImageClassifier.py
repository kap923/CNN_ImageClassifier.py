import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model
import seaborn as sns

'''Images for each category is within subfolders with the category name
Creating a list of subdirectories would give us the Categories'''
DataDir = "\\UCMerced_LandUse\\Images"
Categories = os.listdir(DataDir)
print('Total Number of Categories: ', len(Categories))
print('CategoriesList: ', Categories)

''' HELPER FUNCTIONS'''
#Function to read entire data and assign a category  
ImgData_SingleCategory = []
def Create_ImgData_SingleCategory():
    Img_Size = 200
    for category in Categories:
        #print(category)
        path = os.path.join(DataDir,category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(Img_Size,Img_Size))
                ImgData_SingleCategory.append([new_array,class_num])
            except Exception as e:
                pass

#Function to split Features and Labels
def SplitFeaturesAndLabels(data):
    X = []
    y = []
    for features, labels in data:
        X.append(features)
        y.append(labels)
    X = np.array(X) #Features Vector
    y = np.array(y) #Label Vector
    return X, y

#Defining convolution layer
def conv_layer(inputs,num_filters=64,kernel_size=3,strides=1,activation='relu',batch_normalization=False,pooling=True,dropout=True):
    conv = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
    x = inputs
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    if pooling:
        x = MaxPooling2D(pool_size=(3,3))(x)
    if dropout:
        x = Dropout(0.2)(x)
    return x

#Step #1 --> Read Data and Shuffle the Images
Create_ImgData_SingleCategory()
random.shuffle(ImgData_SingleCategory)

#Step #2 --> Split Feature and Labels, normalize inputs and seperate train and test data
X, y = SplitFeaturesAndLabels(ImgData_SingleCategory)
print(X.shape) #(2100, 200, 200, 3)
print(y.shape) #(2100,)
X = X/255.0 #Normalize the inputs
y = to_categorical(y) #Convert the labels to categorical

X_train = X[:-100,:,:,:]
X_test = X[-100:,:,:,:]
y_train = y[:-100]
y_test = y[-100:]

#Step #4 --> Stack convolution layers and create CNN
N_Labels = 21
input_shape=X.shape[1:]
inputs = Input(shape=input_shape) #Instantiate a Keras Tensor
x = conv_layer(inputs=inputs)
y = conv_layer(inputs=x,num_filters=64,kernel_size=3,strides=1,activation='relu',pooling=True,dropout=True)
y = conv_layer(inputs=y,num_filters=64,kernel_size=3,strides=1,activation='relu',pooling=True,dropout=True)
y = conv_layer(inputs=y,num_filters=32,kernel_size=3,strides=1,activation='relu',pooling=True,dropout=True)
y = Flatten()(y) #This converts our 3D feature maps to 1D future vectors
y = Dense(128,activation='relu')(y)
y = Dropout(0.3)(y)
outputs = Dense(N_Labels,activation='softmax')(y) 

#Step #5 --> Instantiate the Keras model, and run
model = Model(inputs=inputs, outputs=outputs) #Instantiate a Keras Model
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
model.summary()

plot_model(model, to_file=r'\\UCMerced_LandUse\\SingleCategoryModelPlot.png', show_shapes=True, show_layer_names=True)
SingleCategory_Model = model.fit(X_train, y_train, batch_size=10, validation_split=0.1, epochs=100)

# Plot Model performance
plt.figure(figsize=(10,5))
N = 100
plt.plot(np.arange(1, N+1), SingleCategory_Model.history["loss"], label="Train_Loss", linewidth=2)
plt.plot(np.arange(1, N+1), SingleCategory_Model.history["val_loss"], label="Validation_Loss", linewidth=2)
plt.plot(np.arange(1, N+1), SingleCategory_Model.history["accuracy"], label="Train_Accuracy", linewidth=2)
plt.plot(np.arange(1, N+1), SingleCategory_Model.history["val_accuracy"], label="Validation_Accuracy", linewidth=2)
plt.title("Training Loss and Accuracy",fontsize=18, fontweight='bold')
plt.xlabel("Epoch #",fontsize=12, fontweight='bold')
plt.ylabel("Loss/Accuracy",fontsize=12, fontweight='bold')
plt.legend(fontsize='medium',prop=dict(weight='bold'))
plt.savefig('\\UCMerced_LandUse\\SingleCategory_TrainingEpoch.png')

# Use trained model to make prediction of labels
Actual_Test = []
Pred_Test = []
for i in range(len(X_test)):
    pred_data = X_test[i,:,:]
    pred_data = pred_data[np.newaxis,:,:,:]
    #(2100, 100, 100, 3)
    predictions = model.predict(pred_data)
    Actual_Test.append(Categories[np.argmax(y_test[i])])
    Pred_Test.append(Categories[np.argmax(predictions)])

# Model evaluation by confusion matrix
results = confusion_matrix(Actual_Test, Pred_Test) 
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(results, annot=True,linewidths=.5, fmt='d', ax= ax, xticklabels=Categories, yticklabels=Categories)
plt.savefig('\\UCMerced_LandUse\\SingleCategory_ConfusionMatrix.png')