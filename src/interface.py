from multiprocessing import Manager
from tkinter import filedialog as fd
from tkinter import *
from PIL import ImageTk,Image
from tkinter.messagebox import showinfo
import os
from fullPipeline import convertPipelineFull
from tkinter import ttk
import tkinter as tk

picturesDir = os.path.join(os.path.expanduser("~"), "Pictures")

global progressbar

class pipelineWindow(tk.Toplevel):
    def __init__(self, parent, fileName):
        super().__init__(parent)

        self.geometry('300x100')
        self.title('Progess...')

        progressbar = ttk.Progressbar(self, maximum=1000)
        progressbar.place(x=50, y=30, width=200)
        convertPipelineFull(fileName, progressbar)
        progressbar.destroy()
        info = tk.Label(self, text="Finised!", font=("Helvetica", 20))
        info.place(x=50, y=30, width=200)

def makePipelineWindow(fileName):
    window = pipelineWindow(root, fileName)
    window.grab_set()

def resizeInRatio(image, maxSize=(400, 400)):
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

    convertButton = Button(root, text="Convert Image", command=lambda: makePipelineWindow(fileName))    
    convertButton.grid(row=5, column=1, padx=20, pady=(10, 0))



root = Tk()
root.title("Converter")
root.geometry("470x500")
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