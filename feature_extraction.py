import pandas as pd 
import cv2
import numpy as np

def features(img):
    if img.ndim==3 and img.shape[-1]==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim==2:
        gray_img = img
    else:
        print('selected file was nor color neither garyscale')
        return
    gray = gray_img.reshape(-1) # grayscale reshape
    df = pd.DataFrame()
    df['Pixel Intensity'] = gray

    count = 0
    kernel_size=9 # window/kernel size that convolve on image
    for theta in (0, np.pi, np.pi / 4, np.pi / 2): # Theta values define directions
        for sigma in (1, 3):  #Sigma control gaussian distribution or radius of kernel
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelength controls harmonics of frequency
                for gamma in (0.25, 0.5):   #Gamma values defines the length of filter function
                    count += 1
                    gabor_label = 'Gabor' + str(count)
                    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    f_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel) #convolve gabor kernel/window on image
                    Gabor = f_img.reshape(-1)
                    df[gabor_label] = Gabor

    edges = cv2.Canny(gray_img, 100,200)   #gray-image, min and max values
    C_img = edges.reshape(-1)
    df['Canny'] = C_img 

    count=0
    for x in (7,11):
        count+=1
        gaussian_label = 'Gaussian' + str(count)
        gaussian_img = cv2.GaussianBlur(gray_img,(x,x),0)
        gaussian = gaussian_img.reshape(-1)
        df[gaussian_label] = gaussian

    median_img = cv2.medianBlur(gray_img, 3)
    median = median_img.reshape(-1)
    df['Median'] = median
    return df