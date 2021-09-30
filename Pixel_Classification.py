import cv2
import os
import numpy as np
from feature_extraction import features
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import font
from PIL import Image, ImageTk
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
# packages for classifiers, evaluated metrics, and API's #---------------------------------------

import pickle
import joblib

# labeling data-------------------------------------------------------------------------------
def data_labelled():
    global labels
    counts=0
    file_dir_labels = filedialog.askdirectory(title = "Select directory contains labeled data")
    for file_name in (os.listdir(file_dir_labels)):
        file_path = os.path.join(file_dir_labels, file_name)
        print(file_path)
        img = cv2.imread(file_path)
        if img.ndim==3 and img.shape[-1]==3:
            label_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim==2:
            label_img = img
        else:
            print('selected file is nor color neither garyscale image')
        labeled_img = label_img.reshape(-1)
        if counts == 0:
            counts=1
            labels=labeled_img.copy()
        else:
            labels=np.append(labels, labeled_img)

# training data -----------------------------------------------------------------------------
def data_train():
    global model, Val, pro_train, dfeature
    pro_train.start(10)
    root.update_idletasks()
    Val = method_box.get()
    counts=0
    dfeature = pd.DataFrame()
    df_t = pd.DataFrame()
    file_dir_train = filedialog.askdirectory(title = "Select directory contains training images")
    for file_name in (os.listdir(file_dir_train)):
        file_path = os.path.join(file_dir_train, file_name)
        img = cv2.imread(file_path)
        if counts == 0:
            counts +=1
            dfeature=features(img)
        else:
            df_t=features(img)
            dfeature.append(df_t)
    
    if Val == 'Random Forest':
        X_train, X_test, y_train, y_test = train_test_split(dfeature, labels, test_size=0.33, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators = 3, random_state = 42)
        model.fit(X_train, y_train)
    if Val == 'Linear SVM':
        X_train, X_test, y_train, y_test = train_test_split(dfeature, labels, test_size=0.33, random_state=42)
        from sklearn.svm import LinearSVC
        model = LinearSVC(max_iter=200)
        model.fit(X_train, y_train)
    if Val == 'C SVM':
        X_train, X_test, y_train, y_test = train_test_split(dfeature, labels, test_size=0.33, random_state=42)
        from sklearn.svm import SVC
        model = SVC(max_iter=2000)
        model=model.fit(X_train, y_train)
    if Val == 'Ada Boost':
        X_train, X_test, y_train, y_test = train_test_split(dfeature, labels, test_size=0.33, random_state=42) 
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=3, random_state=42)
        model=model.fit(X_train, y_train)
    if Val == 'Gradient Boost':
        X_train, X_test, y_train, y_test = train_test_split(dfeature, labels, test_size=0.33, random_state=42)
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=3, learning_rate=1.0, max_depth=1, random_state=42)
        model=model.fit(X_train, y_train)
    if Val == 'Decision Tree':
        X_train, X_test, y_train, y_test = train_test_split(dfeature, labels, test_size=0.33, random_state=42)
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model=model.fit(X_train, y_train)

    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
    Acu_train = np.around((metrics.accuracy_score(y_train, prediction_train))*100, decimals=1)
    Acu_test = np.around((metrics.accuracy_score(y_test, prediction_test))*100, decimals=1)
    pro_train.stop()
    score = ttk.Entry(Up, width=50)
    score.grid(row=5, column=0, sticky='W',padx=3, pady=3)
    score.insert(tk.END, "Training Classifier Accuracy (train,test): " + str(Acu_train) +", "+ str(Acu_test))
    
# testing data ----------------------------------------------------------------------------------
def data_test():
    global pro_test
    file_dir_test = filedialog.askdirectory(title = "Select directory contains test images")
    file_dir_segmented = filedialog.askdirectory(title = "Select directory to save segmented images")
    num=0
    for file_name in (os.listdir(file_dir_test)):
        file_path = os.path.join(file_dir_test, file_name)
        test_img = cv2.imread(file_path)
        test_feature=features(test_img)
        result = model.predict(test_feature)
        seg_img = result.reshape((test_img.shape[0],test_img.shape[1]))
        num+=1
        pro_test['value']+=100/len(os.listdir(file_dir_test))
        percent.set(str(int((num/len(os.listdir(file_dir_test)))*100))+'%')
        root.update_idletasks()
        plt.imsave(file_dir_segmented + '/segmented_' + file_name, seg_img, cmap ='jet')

def load_classifier():
    file_dir_test = filedialog.askdirectory(title = "Select directory contains test images")
    file_dir_segmented = filedialog.askdirectory(title = "Select directory to save segmented images")
    num=0
    model_filename = filedialog.askopenfilename(initialdir = "/home/fasahat/pCloudDrive/Python_codes/Pixel_classification/",title = "Select Classifier saved in binary file or .sav file format")
    for file_name in (os.listdir(file_dir_test)):
        file_path = os.path.join(file_dir_test, file_name)
        test_img = cv2.imread(file_path)
        test_feature=features(test_img)
        if (os.path.splitext(model_filename))[1] == '.sav':
            loaded_model = joblib.load(model_filename)
            result = loaded_model.predict(test_feature)
        else:
            loaded_model = pickle.load(open(model_filename, 'rb'))
            result = loaded_model.predict(test_feature)
        seg_img = result.reshape((test_img.shape[0],test_img.shape[1]))
        num+=1
        pro_test['value']+=100/len(os.listdir(file_dir_test))
        percent.set(str(int((num/len(os.listdir(file_dir_test)))*100))+'%')
        root.update_idletasks()
        plt.imsave(file_dir_segmented + '/segmented_' + file_name, seg_img, cmap ='jet')

# save model with two diff. API -------------------------------------------------
def save_classifier():
    global API
    API=int(radio_variable.get())
    #print(API)
    model_filename = filedialog.asksaveasfilename(initialdir = "/home/fasahat/pCloudDrive/Python_codes/Pixel_classification/", title = "Save file as")
    if API == 0:
        #Save pickle API
        pickle.dump(model, open(model_filename, 'wb'))
    elif API == 1:
        # joblib API
        model_filename = model_filename+str('.sav')
        joblib.dump(model, model_filename)

#GUI code start from here--------------------------------------------------------
root = tk.Tk()
root.minsize(650, 400)
root.title("Pixel based Image Classification")

#Cap_font = font.Font(name='TkCaptionFont',exists=True,family='Arial Narrow',size=12,slant='roman',weight='normal')
#Cap_font.config
#Head_font= font.Font(name='TkDefaultFont',exists=True,family='Arial Narrow',size=11,slant='roman',weight='normal')
#Head_font.config

menubar = tk.Menu(root)
# file manage bar
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Display")
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)
#-------------------------------------------------------------------------------------
Up = ttk.LabelFrame(root, text="Training Classifiers", width=640, height=180)
Up.grid(row=1, column=0, padx=6, pady=6)

label_button = tk.Button(Up, text="Upload Labeled data", command=data_labelled)
label_button.grid(row=1, column=0, sticky='W', padx=3, pady=3)

list1 = ('Select Classifier','Random Forest', 'Linear SVM', 'C SVM', 'Ada Boost', 'Gradient Boost', 'Decision Tree')
var4 = tk.StringVar()
var4.set(list1[0])
method_box = ttk.Combobox(Up, width = 34, height=3, textvariable=var4, values=list1)
method_box.grid(row=2, column=0,sticky='W', padx=3, pady=3)

train_button2 = tk.Button(Up, text="Train selected Classifier", command=data_train)
train_button2.grid(row=3, column=0,sticky='W', padx=3, pady=3)

pro_train=Progressbar(Up, length=600, orient='horizontal', mode='indeterminate')
pro_train.grid(row=4, column=0, padx=3, pady=3)

score = ttk.Entry(Up, width=50)
score.grid(row=5, column=0, sticky='W',padx=3, pady=3)
score.insert(tk.END, "Training Classifier Accuracy (train,test): 0, 0")
#-------------------------------------------------------------------------------------
bottom = ttk.LabelFrame(root, text="Testing Classifiers", width=640, height=180)
bottom.grid(row=2, column=0, padx=6, pady=6)

train_button = tk.Button(bottom, text="Upload the Classifier", command=load_classifier)
train_button.grid(row=1, column=0,sticky='W', padx=3, pady=3)

seg_button1 = tk.Button(bottom, text="Test Classifiers on Images", command=data_test)
seg_button1.grid(row=1, column=0,sticky='E', padx=3, pady=3)

radio_variable = tk.StringVar()
radio_variable.set("0")
radiobutton1 = ttk.Radiobutton(bottom, text="Use Pickel to save classifier",variable=radio_variable, value="0")
radiobutton2 = ttk.Radiobutton(bottom, text="Use Joblib to save classifier",variable=radio_variable, value="1")
radiobutton1.grid(row=3, column=0, sticky='W', padx=3, pady=3)
radiobutton2.grid(row=3, column=0, sticky='E', padx=3, pady=3)

pro_test=Progressbar(bottom, length=600, orient='horizontal', mode='determinate')
pro_test.grid(row=2, column=0, padx=0, pady=0)
percent = StringVar()
percentlabel=Label(bottom, textvariable=percent).grid(row=2, column=0, padx=0, pady=0)

Upload_button = tk.Button(bottom, text="Save Classifier", command=save_classifier)
Upload_button.grid(row=4, column=0,sticky='W', padx=3, pady=3)

Exit_button = tk.Button(bottom, text='Program Exit', command=root.quit)
Exit_button.grid(row=4,column=0, sticky='E', padx=3, pady=3)
#-------------------------------------------------------------------------------------------

root.iconphoto(False, PhotoImage(file ='/home/fasahat/pCloudDrive/Python_codes/Pixel_classification/Eco-Leafs.png'))
#root.iconbitmap("/home/fasahat/pCloudDrive/Plant_disease_detec/Disease Detection/icon.ico")
root.mainloop()