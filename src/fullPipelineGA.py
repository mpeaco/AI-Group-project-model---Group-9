import os
path = os.path.abspath("vips-dev-8.17/bin")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path
path = os.path.abspath("potrace-1.16.win64")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path

from processing.image_processing import *
from relative_to_absolute import *
from npGeneticAlgorithm import *
from pathJudger import *
from utils.potrace_wrapper import bitmap_to_vector
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk

def convertPipelineFullGA(fileName, progressbar):
    # load image
    #imageLoaded = cv.imread(fileName)
    #imageProcessed = image_processing_opencv(imageLoaded)
    imageLoaded = pyvips.Image.new_from_file(fileName)
    imageProcessed = image_processing_pyvips(imageLoaded)
    # convert image to vector
    cv.imwrite("temp.pbm", imageProcessed)
    newPath = bitmap_to_vector("temp.pbm")
    os.remove("temp.pbm")
    # operate on image to turn the relative paths to absolutes, and then save it in the original location
    writeSvg(relativeToAbsolute(newPath), newPath)
    # open vector for algorithm to run on
    linesIndex, linesOperatable = processFile(newPath)
    for gene in linesOperatable:
        gene[2] = [gene[2], 0]
    bestEver = math.inf
    bestGenes= []
    time1 = time.time()
    patience, patienceCount = 500, 0

    pop = npGeneratePopulation(linesOperatable,50)
    scores = np.array([npJudgeDistance(ind) for ind in pop])
    pop = [pop[i] for i in np.argsort(scores)]

    for x in range(100):
        patienceCount += 1
        pop = npGeneticAlgorithm(pop, 0.2)
        bestScore = npJudgeDistance(pop[0])
        if bestScore < bestEver:
            print(bestScore, x)
            bestEver = bestScore
            bestGenes = pop[0]
            patienceCount = 0
        progressbar.step(1)
        progressbar.update()
        if patienceCount >= patience:
            progressbar.step(10000)
            break

    time2 = time.time()
    print(time2-time1)
    #preview(bestGenes)

    savers = restoreFile(linesIndex, bestGenes)

    fileNameOutput = fileName[:-4] + ".svg"
    writeSvg(savers, fileNameOutput)

    messagebox.showinfo(
        title='Selected File',
        message=("File saved as "+fileName) )
    return