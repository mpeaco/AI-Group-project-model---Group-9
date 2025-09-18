import os
path = os.path.abspath("vips-dev-8.17/bin")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path
path = os.path.abspath("potrace-1.16.win64")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path

from processing.image_processing import *
from relative_to_absolute import *
from pathJudger import *
from utils.potrace_wrapper import bitmap_to_vector
from processing.path_optimisation import *
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
import time

def convertPipelineFullKNN(fileName):
    # load image
    imageLoaded = cv.imread(fileName)
    imageProcessed = image_processing_opencv(imageLoaded)
    # convert image to vector
    cv.imwrite("temp.pbm", imageProcessed)
    newPath = bitmap_to_vector("temp.pbm")
    os.remove("temp.pbm")
    # operate on image to turn the relative paths to absolutes, and then save it in the original location
    writeSvg(relativeToAbsolute(newPath), newPath)
    # open vector for algorithm to run on
    linesIndex, linesOperatable = processFile(newPath)
    time1 = time.time()

    # run the knn function
    linesRunning = []
    linesTemp = []
    temp = [-10, -10]
    for line in linesOperatable:
        #print(line)
        linesTemp.append(CuttingPoint(line[0][0], line[0][1]))
        linesTemp.append(CuttingPoint(line[1][0], line[1][1]))
        if line[0] == temp:
            temp2 = CuttingPath([line2 for line2 in linesTemp])
            linesRunning.append(temp2)
            linesTemp = []
        temp = line[1]
    
    #optimiser = PathOptimizer()
    optimisedPath = optimize_cutting_sequence(linesRunning, method="nearest_neighbor")
    total = 0
    for guy in optimisedPath:
        try:
            total += math.dist([temporempo.x, temporempo.y], [guy.points[0].x, guy.points[0].y])
        except:
            print("fail")
            pass
        #total += guy.length()
        temporempo = guy.points[-1]

    print("total1", total)

    #print(optimisedPath)

    time2 = time.time()
    print("time", time2-time1)
    #preview(bestGenes)
    messagebox.showinfo(
        title='Selected File',
        message=("File saved as "+fileName) )
    return