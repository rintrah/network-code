#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:50:16 2019

Select points of an image and save them in csv file. 

Adapted from Glomschenk's code on stackoverflow 
(https://stackoverflow.com/questions/8590234/capturing-x-y-coordinates-with-python-pil). 

@author: enrique
"""
import tkinter 
from tkinter import filedialog
from PIL import ImageTk, Image
import tifffile 
import numpy as np 
import csv 
#import os 

if __name__=="__main__":
    root = tkinter.Tk() 
    file = open('/home/enrique/points.csv', 'wb')
    writer = csv.writer(file, dialect='excel')
    #setting up a tkinter canvas with scrollbars
    frame = tkinter.Frame(root, bd=2, relief=tkinter.SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = tkinter.Scrollbar(frame, orient=tkinter.HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=tkinter.E+tkinter.W)
    yscroll = tkinter.Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=tkinter.N+tkinter.S)
    canvas = tkinter.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=tkinter.N+tkinter.S+tkinter.E+tkinter.W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=tkinter.BOTH,expand=1)
    
    windows = tkinter.Tk(className="Window")
    
    folder = '/home/enrique/Data/GCaMP6f-Vglut2a--6-days-30Hz-1-min-red-01-17-201900001'
    fileName = 'GCaMP6f-Vglut2a--6-days-30Hz-1-min-red-01-17-201900001_1.tif'
    File = filedialog.askopenfilename(parent=root, initialdir="/home/enrique", title="Select file")
    #image =  Image.open(os.path.join(folder, fileName))
    image = tifffile.imread(File)
    image = (image - image.min())/(image.max() - image.min())
    image = np.uint8(image*255)
    image = ImageTk.PhotoImage(Image.fromarray(image))
    canvas.create_image(0, 0, image=image, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(tkinter.ALL))

    def callback(event):
        writer.writerow([event.x, event.y])
        
    canvas.bind("<Button-1>", callback)
    root.mainloop()