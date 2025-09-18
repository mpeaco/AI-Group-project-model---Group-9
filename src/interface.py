from multiprocessing import Manager
from tkinter import filedialog as fd
from tkinter import *
from PIL import ImageTk,Image
from tkinter.messagebox import showinfo
import os
from fullPipelineKNN import convertPipelineFullKNN
from fullPipelineGA import convertPipelineFullGA
from tkinter import ttk
import tkinter as tk

picturesDir = os.path.join(os.path.expanduser("~"), "Pictures")

global progressbar

class pipelineWindowGA(tk.Toplevel):
    def __init__(self, parent, fileName):
        super().__init__(parent)

        self.geometry('300x100')
        self.title('Progess...')

        progressbar = ttk.Progressbar(self, maximum=10000)
        progressbar.place(x=50, y=30, width=200)
        convertPipelineFullGA(fileName, progressbar)
        progressbar.destroy()
        self.destroy()

def makePipelineWindowGA(fileName):
    window = pipelineWindowGA(root, fileName)
    window.grab_set()

def resizeInRatio(image, maxSize=(300, 300)):
    image.thumbnail(maxSize, Image.LANCZOS)
    return image

def selectFile():
    filetypes = (
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
    )

    fileName = fd.askopenfilename(
        title='Open a file',
        initialdir=picturesDir,
        filetypes=filetypes)
    
    if not fileName:
        return
    
    image = Image.open(fileName)
    image = resizeInRatio(image)
    photo = ImageTk.PhotoImage(image)
    imageLabel.grid(row=4, column=1, padx=20, pady=(10, 0))
    imageLabel.config(image=photo)
    imageLabel.image = photo

    convertButton = Button(root, text="Convert Image KNN", command=lambda: convertPipelineFullKNN(fileName)) 
    convertButton.grid(row=5, column=1, padx=20, pady=(10, 0))

    convertButton = Button(root, text="Convert Image GA", command=lambda: makePipelineWindowGA(fileName))    
    convertButton.grid(row=6, column=1, padx=20, pady=(10, 0))



root = Tk()
root.title("Converter")
root.geometry("600x600")
root.configure(bg="paleturquoise")

welcome_label =Label(root, text="Welcome to The Program.", bg="paleturquoise") 
welcome_label.grid(row=0,column=0,columnspan=6)

frame = LabelFrame(root,borderwidth = 1, padx=50, pady=50, bg="azure")
frame.grid(row=2, column=0,columnspan=6, padx=10, pady=10)

lookupResult= Label(frame)

info = Label(frame, text="Open File Location", bg="azure", font=("Helvetica", 20)).grid(row=0, column=1)

locateButton = Button(root, text="Locate File", command=selectFile)
locateButton.grid(row=3, column=1, padx=20, pady=(10, 0))

imageLabel = Label(root)

root.mainloop()