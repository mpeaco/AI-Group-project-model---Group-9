from pathJudger import *
from relative_to_absolute import writeSvg
import copy
import random as rand
import numpy as np
import math
import time

def npJudgeDistance(linesList):
    starts = linesList[:, 0]
    ends = linesList[:, 1]
    shiftedEnds = np.vstack(([starts[0]], ends[:-1]))
    mask = ~np.all(starts == shiftedEnds, axis=1)
    segDists = np.linalg.norm(starts[mask] - shiftedEnds[mask], axis=1)

    #lineDists = np.linalg.norm(ends - starts, axis=1)

    return np.sum(segDists)# + np.sum(lineDists)

def npGeneratePopulation(genes, popSize):
    genes = np.array(genes)
    output = np.empty((popSize, *np.shape(genes)))
    #output[0] = copy.deepcopy(genes)
    index = 0
    while index != popSize:
        tempnum = rand.randint(1, len(genes))
        temp1 = genes[:tempnum]
        temp2 = genes[tempnum:]
        temp3 = np.concatenate((temp2, temp1))
        for x in range(int(len(temp3)/10)):
            point = rand.choice(range(len(temp3)-1))
            temp3[[point,point+1]] = temp3[[point+1, point]]
            temp3 = npMutateRotate(temp3)
        output[index] = (copy.deepcopy(temp3))
        index += 1
    
    return output

def npMutateFlip(individual):
    flipped = individual.copy()
    flipped[:, [0,1]] = flipped[:, [1,0]]
    return flipped
    
def npMutateRotate(individual):
    output = individual
    numb = rand.randint(0,len(individual))
    return np.concatenate((output[numb:], output[:numb]))

def npGeneticAlgorithm(population, mutRate):
    children = np.empty(np.shape(population))
    length = len(population)
    numGenes = len(population[0])

    randParents = np.random.choice(length, size=(length, 2))
    randCrossover = np.random.choice(numGenes, size=(length, 2))
    randCrossover = np.sort(randCrossover, axis=1, kind="quicksort")

    for j in range(length):
        parent1, parent2 = population[randParents[j][0]], population[randParents[j][1]]
        child = np.empty(np.shape(parent1))

        start, end = randCrossover[j]
        child[start:end] = parent1[start:end]
        k = 0
        usedGenes = {gene[2][0] for gene in child[start:end]}
        for gene in parent2:
            while start <= k < end:
                k += 1
            if gene[2][0] not in usedGenes:
                child[k] = gene
                k += 1

        if rand.random() <= mutRate:
            child = npMutateFlip(child)
        if rand.random() <= mutRate:
            child = npMutateRotate(child)
        # swap 2 adjacent points
        if rand.random() <= mutRate:
            point = rand.choice(range(len(child)-1))
            child[[point,point+1]] = child[[point+1, point]]
        # swap 2 random points
        if rand.random() <= mutRate:
            points = rand.choices(range(len(child)-1), k=2)
            child[[points[0],points[1]]] = child[[points[1], points[0]]]
        children[j] = child

    combined = np.concatenate((children, population[:int(np.round(length/10))]))
    scores = np.array([npJudgeDistance(ind) for ind in combined])
    bestInd = np.argsort(scores)[:length]
    output = combined[bestInd]

    return output


def main():
    orig, lines = processFile("outputSun.svg")

    bestEverEverEver = [math.inf, 0]

    for gene in lines:
        gene[2] = [gene[2], 0]
    mut = 0
    for mutar in range(5):
        bestEverEver = math.inf
        mut += 0.1
        time1 = time.time()
        for p in range(10):

            pop = npGeneratePopulation(lines,50)
            bestScoreEver = math.inf
            for k in range(1000):
                pop = npGeneticAlgorithm(pop, mut)
                bestScore = npJudgeDistance(pop[0])
                if bestScore < bestScoreEver:
                    #print(bestScore, k)
                    bestScoreEver = bestScore

            if bestScoreEver < bestEverEver:
                bestEverEver = bestScoreEver
                bestGeneEver = pop[0]

        time2 = time.time()
        print("mut ",mut," time ",(time2-time1)/15)

        print(bestEverEver)
        if bestEverEver < bestEverEverEver[0]:
            bestEverEverEver = [bestEverEver, mut]
    print(bestEverEverEver)
    preview(bestGeneEver)


if __name__=="__main__":
    main()