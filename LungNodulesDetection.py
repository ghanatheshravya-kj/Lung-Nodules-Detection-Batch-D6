from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import numpy as np
import os
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import cv2
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras
import pickle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

main = tkinter.Tk()
main.title("False-Positive Reduction on Lung Nodules Detection in Chest Radiographs by Ensemble of Convolutional Neural Networks")
main.geometry("1300x900")

global classifier
global filename
global X,Y
global imagePaths
global bboxes
global model

def upload():
  global filename
  global X,Y
  global imagePaths
  global bboxes
  filename = filedialog.askdirectory(initialdir = ".")
  X = np.load("model/img.txt.npy")
  Y = np.load("model/labels.txt.npy")
  imagePaths = np.load('model/files.txt.npy')
  bboxes = np.load('model/bbox.txt.npy')

  text.delete('1.0', END)
  text.insert(END,filename+' train images dataset Loaded\n\n')
  text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
  pathlabel.config(text=filename+" loaded")

  img = cv2.imread(imagePaths[0])
  bb = bboxes[0]  
  xmin = int(bb[0] * 2048)
  ymin = int(bb[1] * 2048)
  xmax = int(bb[2] * 2048)
  ymax = int(bb[3] * 2048)

  cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 0, 255), 3)
  cv2.imwrite("test.jpg",img)
  img = cv2.imread("test.jpg")
  img = cv2.resize(img,(500,500))
  cv2.imshow("a",img)
  cv2.waitKey(0)

    

def loadModel():
    global X,Y
    global classifier
    global model
    if os.path.exists('model/model.h5'):
        classifier = load_model('model/model.h5')
        with open('model/cnnmodel.json', "r") as json_file:
          loaded_model_json = json_file.read()
          model = model_from_json(loaded_model_json)
        json_file.close()
        model.load_weights("model/cnn_weights.h5")
        model._make_predict_function()   
        text.insert(END,'CNN Models Generated Successfully. See black console for CNN layers\n')
    else:
        split = train_test_split(X, Y, bboxes, imagePaths, test_size=0.20, random_state=42)
        (trainImages, testImages) = split[:2]
        (trainLabels, testLabels) = split[2:4]
        (trainBBoxes, testBBoxes) = split[4:6]
        (trainPaths, testPaths) = split[6:]
        vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(100, 100, 3)))
        vgg.trainable = False
        flatten = vgg.output
        flatten = Flatten()(flatten)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(512, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(6, activation="softmax", name="class_label")(softmaxHead)
        model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))
        losses = {
          "class_label": "categorical_crossentropy",
          "bounding_box": "mean_squared_error",
        }
        lossWeights = {
          "class_label": 1.0,
          "bounding_box": 1.0
        }
        opt = Adam(lr=1e-4)
        model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        print(model.summary())
        trainTargets = {
          "class_label": trainLabels,
          "bounding_box": trainBBoxes
        }
        testTargets = {
          "class_label": testLabels,
          "bounding_box": testBBoxes
        }
        hist = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=20, verbose=1)
    f = open('model/history.pckl', 'rb')
    model_acc = pickle.load(f)
    f.close()
    accuracy = model_acc['accuracy']
    loss = model_acc['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Sensitivity/False Rate')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Sensitivity', 'False Rate'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('ECNN Sensitivity & False Rate Graph on Lung Nodules')
    plt.show()
    
def detection():
    global classifier
    global model
    name = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=name+" loaded")
    image = load_img(name, target_size=(100, 100))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    (boxPreds, labelPreds) = classifier.predict(image)
    boxPreds = boxPreds[0]
    print(boxPreds)
    xmin = int(boxPreds[0] * 2048)
    ymin = int(boxPreds[1] * 2048)
    xmax = int(boxPreds[2] * 2048)
    ymax = int(boxPreds[3] * 2048)
    predict = np.argmax(labelPreds, axis=1)
    image = cv2.imread(name)
    img = cv2.resize(image, (100,100))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,100,100,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = model.predict(img)
    predict = np.argmax(preds)
    print(str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+str(predict))
    msg = 'No Lung Cancer Detected'
    img = cv2.imread(name)
    if predict > 0 and predict <= 5:
      msg = "Lung Cancer Detected"
      cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 0, 255), 3)
      text.insert(END,"Lung Cancer Nodules Bounding Box Location Predicted : "+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+"\n")
    cv2.imwrite("test.jpg",img)
    img = cv2.imread("test.jpg")
    img = cv2.resize(img,(500,500))
    cv2.putText(img, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)

def exit():
    global main
    main.destroy()
  

font = ('times', 16, 'bold')
title = Label(main, text='False-Positive Reduction on Lung Nodules Detection in Chest Radiographs by Ensemble of Convolutional Neural Networks',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
loadButton = Button(main, text="Upload JSRT Lung Nodules Dataset", command=upload)
loadButton.place(x=50,y=200)
loadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


uploadButton = Button(main, text="Generate & Load 3 CNN Models", command=loadModel)
uploadButton.place(x=50,y=300)
uploadButton.config(font=font1)

uploadButton = Button(main, text="Upload Test Image & Detect Lung Nodules", command=detection)
uploadButton.place(x=50,y=350)
uploadButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)

text=Text(main,height=20,width=70)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=200)
text.config(font=font1) 

main.config(bg='chocolate1')
main.mainloop()
