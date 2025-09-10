import os
path = os.path.abspath("vips-dev-8.17/bin")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path
path = os.path.abspath("potrace-1.16.win64")
os.environ["PATH"] = os.environ['PATH'] + os.pathsep + path

from image_processing import *
from relative_to_absolute import *
from npGeneticAlgorithm import *
from pathJudger import *
from potrace_wrapper import bitmap_to_vector
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk

def convertPipelineFull(fileName, progressbar):
    # convert image to vector
    newPath = bitmap_to_vector(fileName)
    # operate on image to turn the relative paths to absolutes, and then save it in the original location
    writeSvg(relativeToAbsolute(newPath), newPath)
    # open vector for algorithm to run on
    linesIndex, linesOperatable = processFile(newPath)
    for gene in linesOperatable:
        gene[2] = [gene[2], 0]
    bestEver = math.inf
    bestGenes= []
    time1 = time.time()
    pop = npGeneratePopulation(linesOperatable,50)
    for x in range(1000):
        pop = npGeneticAlgorithm(pop, 0.3)
        bestScore = npJudgeDistance(pop[0])
        if bestScore < bestEver:
            print(bestScore, x)
            bestEver = bestScore
            bestGenes = pop[0]
        progressbar.step(1)
        progressbar.update()
    time2 = time.time()
    print(time2-time1)
    #preview(bestGenes)
    messagebox.showinfo(
        title='Selected File',
        message=("File saved as "+fileName) )
    return