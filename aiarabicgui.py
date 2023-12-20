
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image,ImageDraw
import cv2
from ctypes import windll, byref, sizeof, c_int



width,height=500,500
clf =tf.keras.models.load_model('mnis.h5')
win=tk.Tk()
win.configure(bg="black")
win.title("farah project")
win.update()
HWND = windll.user32.GetParent(win.winfo_id())
COLOR_1 =0xcc16ba 
COLOR_2 =0xfafafa# color should be in hex order: 0x00bbggrr
windll.dwmapi.DwmSetWindowAttribute(HWND, 35, byref(c_int(COLOR_1)), sizeof(c_int))
windll.dwmapi.DwmSetWindowAttribute(HWND, 36, byref(c_int(COLOR_2)), sizeof(c_int))
font_btn='Helvetica 20 bold'




def event_function(event):
    x=event.x       #x coordinate of mouse pointer
    y=event.y       #y coordinate of mouse pointer
    x1=x-20
    y1=y-20
    x2=x+20
    y2=y+20
    canvas.create_oval((x1,y1,x2,y2),fill='black')
    img_draw.ellipse((x1,y1,x2,y2),fill='white')


def clear():
    global img,img_draw
    canvas.delete('all')
    img=Image.new('RGB',(width,height),(0,0,0))
    img_draw=ImageDraw.Draw(img)
   

def predict():
    global img

    img_array=np.array(img) #converting to numpy array
    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY) #converting into a gray image
    img_array1=cv2.resize(img_array.T,(28,28)) #resizing into 28x28    # T to be compatible with the train data
    img_array=np.reshape(img_array1,(1,784))  
    img_array=img_array/255.0
    result=clf.predict(img_array)
    plt.imshow(img_array1.T, cmap='gray',)
    plt.show()
    print(f"This digit is probably a {np.argmax(result)}")
    
canvas=tk.Canvas(win,width=500, height=500, bg = "white", highlightthickness=40, highlightbackground="black", cursor="cross")
canvas.grid(row=1,column=0,columnspan=4)

canvas.bind('<B1-Motion>',event_function)
img=Image.new('RGB',(width,height),(0,0,0))
img_draw=ImageDraw.Draw(img)


button_predict=tk.Button(win,text='PREDICT',background="#FF00FF",fg='white',font=font_btn,command=predict,borderwidth = '4')
button_predict.grid(row=2,column=1)

button_clear=tk.Button(win,text='CLEAR',background="#FF00FF",fg='white',font=font_btn,command=clear,borderwidth = '4')
button_clear.grid(row=2,column=2)


win.mainloop()