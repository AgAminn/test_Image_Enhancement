
import cv2
from PIL import Image
from numpy.fft import fft
import numpy as np
    
def img_enhance_01 (path_im):
    # read the original image
    #img = Image.open('orig_chest_xray.tif')
    img = Image.open(path_im).convert('L')
    # compute fft shoft of the original img
    npFFT = np.fft.fft2(img) # Calculate FFT
    npFFTS = np.fft.fftshift(npFFT)  # Shift the FFT to center it
    
    # compute HFE filter using Guassian High-pass filter
    print('this shape is ',npFFTS.shape)
    (P, Q) = npFFTS.shape
    H = np.zeros((P,Q))
    D0 = 40
    for u in range(P):
        for v in range(Q):
            H[u, v] = 1.0 - np.exp(- ((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2) / (2 * (D0 ** 2)))
    k1 = 0.5 ; k2 = 0.75
    HFEfilt = k1 + k2 * H # Apply High-frequency emphasis

    #Apply the HFE filter  (by multiplying HFE with the FFT of original image
    HFE = HFEfilt * npFFTS

    #Perform the inverse Fourier transform and generate an image to view the results.
    """
    Implement 2D-FFT algorithm

    Input : Input Image
    Output : 2D-FFT of input image
    """
    def fft2d(image):
        # 1) compute 1d-fft on columns
        
        fftcols = np.array([fft(row) for row in image]).transpose()

        # 2) next, compute 1d-fft on in the opposite direction (for each row) on the resulting values
        return np.array([fft(row) for row in fftcols]).transpose()


    #Perform IFFT (implemented here using the np.fft function)
    HFEfinal = (np.conjugate(fft2d(np.conjugate(HFE)))) / (P * Q)
    return HFEfinal

def img_enhance_02(img_o):
    #Calculate Probability density function 
    # (you can also use the in-built np.histogram function)
    #calculate pdf
    #cv2.imread(path_im,0)
    W,H = img_o.shape
    numofpixels = W * H
    freq = {}
    probf = {}
    for i in range(1, W):
        for j in range(1, H):
            value = img_o[i, j]
            Lk = list(freq.keys() )
            if value not in Lk :
                freq[value]= 0
            freq[value] += 1
            probf[value] = freq[value] / numofpixels
    print(' P1')
    #Calculate Cumulative Density function
    sum = 0
    num_bins = 255
    cum = {}
    probc = {}
    output = {}
    # calculate CDF
    Lk = list(freq.keys()) 
    for i in Lk:
        sum = sum + freq[i]
        cum[i] = sum
        probc[i] = cum[i] / numofpixels
        output[i] = round(probc[i] * num_bins)
    
    # Final Histogram Equalization image
    histImg = np.zeros((W,H),dtype=np.uint8)
    for i in range(1, W):
        for j in range(1, H):
            histImg[i,j] = output[img_o[i, j]]
    return histImg

if __name__ == '__main__':
    img_path= r'test_imgs\xray.jpg'

    im_orig = cv2.imread(img_path)
    cv2.imshow('img orig',im_orig)
    img1 = img_enhance_01(img_path)
    cv2.imshow('test 1',img1.astype(np.uint8))

    img1 = cv2.imread(img_path,0)
    img2 = img_enhance_02(img1)
    print(img2.shape)
    cv2.imshow('test 2',img2)
    cv2.waitKey(0)
