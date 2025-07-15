from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from copy import copy
import ctypes
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL.Image')

original_image = None
noisy_image = None
result_image = None

def open_image():
    global original_image
    file_path = filedialog.askopenfilename(title="Open Image",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        original_image = cv2.imread(file_path)
        display_image(original_image, canvas1)

def display_image(image, canvas):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, (350, 180))  
    img = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(img)
    canvas.image = img_tk  
    canvas.create_image(0, 0, anchor=NW, image=img_tk)

def convert_gray():
    global original_image
    if original_image is not None:
        gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        display_image(gray_image, canvas1)

def convert_orig():
    global original_image
    if original_image is not None:
        display_image(original_image, canvas1)

def salt_and_pepper():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = original_image.copy()
        no_row,no_col = noisy_image.shape[:2]

        for i in range(0,int( .05*no_col*no_row)):
            x = np.random.randint(0, no_col)
            y = np.random.randint(0, no_row)

            noisy_image[y][x] = 255


        for i in range(0, int(.02*no_col*no_row)):
            x = np.random.randint(0, no_col)
            y = np.random.randint(0, no_row)

            noisy_image[y][x] = 0    

    display_image(noisy_image, canvas2)

def gaussian_noise():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = original_image.copy()
        gauss = np.random.normal(loc=0, scale=3, size=original_image.shape).astype('uint8')
        noisy_image = cv2.add(noisy_image, gauss)
        display_image(noisy_image, canvas2)

def poisson_noise():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = original_image.copy()
        noisy_image = np.random.poisson(original_image).astype('uint8')
        display_image(noisy_image, canvas2)

def brightness_adjustment():
    global original_image, result_image
    if original_image is not None:
        result_image = original_image.copy()
        cv2.convertScaleAbs(result_image, alpha=1.0, beta=50)     
        display_image(result_image, canvas3)

def contrast_adjustment():
    global original_image, result_image
    if original_image is not None:
        result_image = original_image.copy()
        cv2.convertScaleAbs(result_image, alpha=1.5, beta=0)     
        display_image(result_image, canvas3)

def histogram():
    global original_image, result_image
    if original_image is not None:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        hist_img = np.full((256, 256, 3), 255, dtype=np.uint8)

        for x in range(256):
            y = int(hist[x])
            cv2.line(hist_img, (x, 255), (x, 255 - y), (0, 0, 0), 1)
            cv2.rectangle(hist_img, (x, 255 - y), (x, 255), (255, 0, 0), -1)
        result_image = hist_img
        display_image(result_image, canvas3)
       
def histogram_equalization():
    global original_image, result_image
    if original_image is not None:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        result_image = cv2.equalizeHist(gray_image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)  
        display_image(result_image, canvas3)

def low_pass_filter():
    global original_image, noisy_image
    if original_image is not None :
        noisy_image = original_image.copy()
        noisy_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
        display_image(noisy_image, canvas3)

def high_pass_filter():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = original_image.copy()
        kernel = np.array([[0, -1, 0],
                           [-1,  4, -1],
                           [0, -1, 0]])
        noisy_image = cv2.filter2D(noisy_image, -1, kernel)
        display_image(noisy_image, canvas3)

def median_filter():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = original_image.copy()
        noisy_image = cv2.medianBlur(noisy_image, 5)
        display_image(noisy_image, canvas3)

def average_filter():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = original_image.copy()
        noisy_image = cv2.blur(noisy_image, (3, 3))
        display_image(noisy_image, canvas3)

def laplacian_filter():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = cv2.Laplacian(original_image, cv2.CV_64F)
        noisy_image = cv2.convertScaleAbs(noisy_image)
        display_image(noisy_image, canvas3)

def gaussian_filter():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = cv2.GaussianBlur(original_image, (3, 3), 0)
        display_image(noisy_image, canvas3)

def sobel_v():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
        noisy_image = cv2.convertScaleAbs(noisy_image)
        display_image(noisy_image, canvas3)

def sobel_h():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)
        noisy_image = cv2.convertScaleAbs(noisy_image)
        display_image(noisy_image, canvas3)

def prewitt_v():
    global original_image, noisy_image
    if original_image is not None:
        kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])
        noisy_image = cv2.filter2D(original_image, -1, kernel)
        display_image(noisy_image, canvas3)

def prewitt_h():
    global original_image, noisy_image
    if original_image is not None:
        kernel = np.array([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]])
        noisy_image = cv2.filter2D(original_image, -1, kernel)
        display_image(noisy_image, canvas3)

def log_filter():
    global original_image, noisy_image
    if original_image is not None:
        blur = cv2.GaussianBlur(original_image, (3, 3), 0)
        noisy_image = cv2.Laplacian(blur, cv2.CV_64F)
        noisy_image = cv2.convertScaleAbs(noisy_image)
        display_image(noisy_image, canvas3)

def canny_edge():
    global original_image, noisy_image
    if original_image is not None:
        noisy_image = cv2.Canny(original_image, 100, 200)
        display_image(noisy_image, canvas3)

def zero_cross():
    global original_image, noisy_image
    if original_image is not None:
        laplacian = cv2.Laplacian(original_image, cv2.CV_64F)
        noisy_image = np.uint8(np.absolute(laplacian))
        _, noisy_image = cv2.threshold(noisy_image, 10, 255, cv2.THRESH_BINARY)
        display_image(noisy_image, canvas3)

def thicken():
    global original_image, noisy_image
    if original_image is not None:
        kernel = np.ones((3, 3), np.uint8)
        noisy_image = cv2.dilate(original_image, kernel, iterations=1)
        display_image(noisy_image, canvas3)

def skeletonize():
    global original_image, noisy_image
    if original_image is not None:
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            open_ = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open_)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break

        noisy_image = skel
        display_image(noisy_image, canvas3)

def thinning():
    global original_image, noisy_image
    if original_image is not None:
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        noisy_image = cv2.erode(img, None, iterations=1)
        display_image(noisy_image, canvas3)

def hough_lines():
    global original_image

    if original_image is not None:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)

        result_image = original_image.copy()

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        display_image(result_image, canvas3)

        display_image(edges, canvas2)  


def hough_circles():
    global original_image
    if original_image is None:
        return

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)

    accumulator = cv2.Canny(blurred, 100, 200)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    result = original_image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)


    display_image(accumulator, canvas2)

    display_image(result, canvas3)


def apply_morph(operation, kernel_type='rect', size=3):
    global original_image, result_image
    if original_image is None:
        return
    
    if kernel_type == 'rect':
        kernel = np.ones((size, size), np.uint8)
    elif kernel_type == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif kernel_type == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:  
        kernel = np.ones((size, size), np.uint8)
    
    if operation == 'dilate':
        result_image = cv2.dilate(original_image, kernel)
    elif operation == 'erode':
        result_image = cv2.erode(original_image, kernel)
    elif operation == 'open':
        result_image = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        result_image = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)
    
    display_image(result_image, canvas3)

def dilate(): apply_morph('dilate')
def erode(): apply_morph('erode')
def open(): apply_morph('open')
def close(): apply_morph('close')

def save_result():
    global result_image
    if result_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, result_image)


root = Tk()
root.geometry("1000x750+300+0")
root.resizable(True,True)
root.title("Image Processing Project")
root.config(background="white")
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(BASE_DIR, 'project_icon.ico')
    root.iconbitmap(True, icon_path)
    if sys.platform.startswith('win'):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u'mycompany.myproduct.subproduct.version')
except Exception as e:
    print("Error setting icon:", e)

lf1 =LabelFrame(root,
                text="Load image"  ,   
                font=('Arial', 8, 'bold'),fg='red',width=180,height=100,  bg='white')  
            
lf1.place(x=10,y=10)

btn = Button(lf1, text="Open..", width=16, height=2, fg="black", bg="white", cursor="hand2",command = open_image)
btn.place(x=30, y=10)



lf2 =LabelFrame(root,
                text="Convert"  ,   
                font=('Arial', 8, 'bold'),
        fg='red',width=190,height=100,
        bg='white')  

lf2.place(x=200,y=10)

r1= Radiobutton(lf2,text="Default color",value=1,fg="black",bg="white",cursor="hand2",command=convert_orig)
r1.place(x=10,y=5)

r2= Radiobutton(lf2,text="Gray color",value=2,fg="black",bg="white",cursor="hand2",command=convert_gray)
r2.place(x=10,y=40)





lf3 =LabelFrame(root,
                text="Add noise"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
            highlightbackground='yellow',highlightcolor='yellow',
        fg='red',width=190,height=100,
        bg='white')  

lf3.place(x=400,y=10)

r3= Radiobutton(lf3,text="Salt & Pepper noise",value=3,fg="black",bg="white",cursor="hand2",command=salt_and_pepper)
r3.place(x=10,y=5)

r4= Radiobutton(lf3,text="Gaussian noise",value=4,fg="black",bg="white",cursor="hand2",command=gaussian_noise)
r4.place(x=10,y=30)

r5= Radiobutton(lf3,text="Poisson noise",value=5,fg="black",bg="white",cursor="hand2",command=poisson_noise)
r5.place(x=10,y=50)


lf4 =LabelFrame(root,
                text="Point Transform Op's"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=580,height=180,
        bg='white')  

lf4.place(x=10,y=120)

bt1 = Button(lf4, text="Brightness adjustment",fg="black",bg="White",cursor="hand2",width=18,height=1,command=brightness_adjustment)
bt1.place(x=10,y=10)

bt2 = Button(lf4, text="Contrast adjustment",fg="black",bg="White",cursor="hand2",width=18,height=1,command=contrast_adjustment)
bt2.place(x=150,y=50)

bt3 = Button(lf4, text="Histogram",fg="black",bg="White",cursor="hand2",width=18,height=1,command=histogram)
bt3.place(x=270,y=90)

bt4 = Button(lf4, text="Histogram Equalization",fg="black",bg="White",cursor="hand2",width=18,height=1,command=histogram_equalization)
bt4.place(x=410,y=130)



lf5 =LabelFrame(root,
                text="Local Transform Op's"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=580,height=180,
        bg='white')  

lf5.place(x=10,y=310)

bt5 = Button(lf5, text="Low pass filter",fg="black",bg="White",cursor="hand2",width=22,height=1,command=low_pass_filter)
bt5.place(x=10,y=10)

bt6 = Button(lf5, text="High pass filter",fg="black",bg="White",cursor="hand2",width=22,height=1,command=high_pass_filter)
bt6.place(x=10,y=50)

bt7 = Button(lf5, text="Median filtering (gray image)",fg="black",bg="White",cursor="hand2",width=22,height=1,command=median_filter)
bt7.place(x=10,y=90)

bt8 = Button(lf5, text="Averaging filtering",fg="black",bg="White",cursor="hand2",width=22,height=1,command=average_filter)
bt8.place(x=10,y=130)

lf6 =LabelFrame(lf5,
                text="Edge detection filters"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=400,height=150,
        bg='white')  

lf6.place(x=175,y=5)

r6= Radiobutton(lf6,text="Laplacian filter",value=6,fg="black",bg="white",cursor="hand2",command=laplacian_filter)
r6.place(x=5,y=5)

r7= Radiobutton(lf6,text="Gaussain filter",value=7,fg="black",bg="white",cursor="hand2",command=gaussian_filter)
r7.place(x=100,y=5)

r8= Radiobutton(lf6,text="V. Sobel",value=8,fg="black",bg="white",cursor="hand2",command=sobel_v)
r8.place(x=200,y=5)

r9= Radiobutton(lf6,text="H. Sobel",value=9,fg="black",bg="white",cursor="hand2",command=sobel_h)
r9.place(x=290,y=5)

r9= Radiobutton(lf6,text="V. Perwitt",value=10,fg="black",bg="white",cursor="hand2",command=prewitt_v)
r9.place(x=5,y=55)

r10= Radiobutton(lf6,text="H. Perwitt",value=11,fg="black",bg="white",cursor="hand2",command=prewitt_h)
r10.place(x=100,y=55)

r11= Radiobutton(lf6,text="LOG",value=12,fg="black",bg="white",cursor="hand2",command=log_filter)
r11.place(x=200,y=55)

r12= Radiobutton(lf6,text="Canny method",value=13,fg="black",bg="white",cursor="hand2",command=canny_edge)
r12.place(x=290,y=55)

r13= Radiobutton(lf6,text="Zero Cross",value=14,fg="black",bg="white",cursor="hand2",command=zero_cross)
r13.place(x=5,y=105)

r14= Radiobutton(lf6,text="Thicken",value=15,fg="black",bg="white",cursor="hand2",command=thicken)
r14.place(x=100,y=105)

r15= Radiobutton(lf6,text="skeleton",value=16,fg="black",bg="white",cursor="hand2",command=skeletonize)
r15.place(x=200,y=105)

r16= Radiobutton(lf6,text="thining",value=17,fg="black",bg="white",cursor="hand2",command=thinning)
r16.place(x=290,y=105)



lf7 =LabelFrame(root,
                text="Global Transform Op's"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=250,height=180,
        bg='white')  

lf7.place(x=10,y=500)

bt9 = Button(lf7, text="Line detection using Hough Transform ",fg="black",bg="White",cursor="hand2",width=32,height=1,command=hough_lines)
bt9.place(x=10,y=30)

bt10 = Button(lf7, text="Circles detection using Hough Transform ",fg="black",bg="White",cursor="hand2",width=32,height=1,command=hough_circles)
bt10.place(x=10,y=110)


lf8 =LabelFrame(root,
                text="Morphological Op's"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=320,height=180,
        bg='white')  

lf8.place(x=270,y=500)

bt11 = Button(lf8, text="Dilation ",fg="black",bg="White",cursor="hand2",width=16,height=1,command=dilate)
bt11.place(x=10,y=20)

bt12 = Button(lf8, text="Erosion ",fg="black",bg="White",cursor="hand2",width=16,height=1,command=erode)
bt12.place(x=10,y=55)

bt13 = Button(lf8, text="Close ",fg="black",bg="White",cursor="hand2",width=16,height=1,command=close)
bt13.place(x=10,y=90)

bt14 = Button(lf8, text="Open ",fg="black",bg="White",cursor="hand2",width=16,height=1,command=open)
bt14.place(x=10,y=125)

lab = Label(lf8, text="Choose type of Kernal",fg="yellowgreen",bg="white",cursor="hand2", width=17,height=1)
lab.place(x=160,y=40)

cmbol = ttk.Combobox(lf8,values=['Arbitrary', 'diamond', 'disk', 'line', 'octagon', 'pair', 'periodic', 
'line', 'rectangle', 'square'],width=22,height=3, state = 'readonly')
cmbol.set("Arbitrary")
cmbol.place(x=160,y=70)


bt15 = Button(root, text="Save Result image",fg="black",bg="White",cursor="hand2",width=16,height=2,command=save_result)
bt15.place(x=60,y=700)

bt16 = Button(root, text="Exit",fg="black",bg="White",cursor="hand2",width=16,height=2,command=root.destroy)
bt16.place(x=400,y=700)


lf9 =LabelFrame(root,
                text="Original image"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=370,height=220,
        bg='white')  

lf9.place(x=610,y=10)

canvas1= Canvas(lf9, width=350, height=180, bg="white")
canvas1.place(x=5,y=5)


lf10 =LabelFrame(root,
                text="after noise adding"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=370,height=220,
        bg='white')  

lf10.place(x=610,y=260)

canvas2= Canvas(lf10, width=350, height=180, bg="white")
canvas2.place(x=5,y=5)

lf11 =LabelFrame(root,
                text="Result"  ,   
                font=('Arial', 8, 'bold'),borderwidth=1,
        fg='red',width=370,height=220,
        bg='white')  

lf11.place(x=610,y=510)

canvas3= Canvas(lf11, width=350, height=180, bg="white")
canvas3.place(x=5,y=5)




root.mainloop()


