import random
import numpy as np
import sys
import matplotlib.pyplot as plt
#from matplotlib import style
import math
import operator
import pandas as pd

#style.use('fivethirtyeight')
'''
this function  will get in the number of genes in the chromosomes(length) and the number
of ones (number) and will return the random distribution of the ones
'''

# create a random population. the bits argument defines the number of bits (genes) each chromosome will consist of
# popSize will determine the number of chromosomes and 'onesMax' determines the maximum number of ones for each chromosome
def random_B_initialization(population, bits, popSize, onesMin=0, onesMax=0):
    #first we choose randomly how many ones each chromosome will have
    #and then their positions
    if onesMax != 0 and onesMin != 0:
        for ite in range(popSize):
            i = random.randrange(onesMin, onesMax + 1)
            #print("the random number = ", i, "\n")
            a = random.sample(range(0, bits), i)
            #print(a, '\n')
            for pos in a:
                population[ite, pos] = 1

    elif onesMax != 0:
        for ite in range(popSize):
            i = random.randrange(0, onesMax + 1)
            #print("the random number = ", i, "\n")
            a = random.sample(range(0, bits), i)
            #print(a, '\n')
            for pos in a:
                population[ite, pos] = 1

    elif onesMin != 0:
        for ite in range(popSize):
            i = random.randrange(onesMin, bits + 1)
            #print("the random number = ", i, "\n")
            a = random.sample(range(0, bits), i)
            #print(a, '\n')
            for pos in a:
                population[ite, pos] = 1

    else:

        for ite in range(popSize):
            i = random.randrange(0, bits + 1)
            #print("the random number = ", i, "\n")
            a = random.sample(range(0, bits), i)
            #print(a, '\n')
            for pos in a:
                population[ite, pos] = 1

    return population

def fitness(s, p):
    global want_sum, want_prod
    error = abs(want_prod - p) + abs(want_sum - s) * 10
    return 1 / (error + 1)


def evaluation(population, number_of_chroms):
    global genes
    values = np.zeros((number_of_chroms,1))

    if number_of_chroms == 1:
        sum = 0
        prod = 1
        for j in range(0, genes):
            if population[j] == 0:
                sum += (j + 1)
            else:
                prod *= (j + 1)

        return sum,prod

    else:
        for i in range(0, number_of_chroms):
            sum = 0
            prod = 1
            for j in range(0, genes):
                if population[i, j] == 0:
                    sum += (j + 1)
                else:
                    prod *= (j + 1)

            values[i] = fitness(sum, prod)   
    return values

def evaluationOfOne(chromosome, genes):
    sum = 0
    prod = 1
    for j in range(0, genes):
        if chromosome[j] == 0:
            sum += (j + 1)
        else:
            prod *= (j + 1)

    print("the best sum is ", sum, " and the best product is ", prod)    


def avgValue(size):
    global values
    sum = 0
    for i in values:
        sum += i

    return sum / size

def generalEvaluation(population2Eval,popSize, genes):
    values = np.zeros((popSize,1))
    for i in range(0, popSize):
        sum = 0
        prod = 1
        for j in range(0, genes):
            if population2Eval[i, j] == 0:
                sum += (j + 1)
            else:
                prod *= (j + 1)
        
        values[i] = fitness(sum, prod)

    return values



# findMaxParent will accept a 2 x ways array of parents' indices and will return a 2x1 array. The best parents
# notice that randomCouple array contains the positions of the parents, not their values
def findMaxParent(randomCouple, ways):
    global values

    max1 = randomCouple[0,0]
    max2 = randomCouple[1,0]
    for i in range(1, ways):
        v1 = values[int(max1)]
        v2 = values[int(max2)]
        vTemp1 = values[int(randomCouple[0,i])]
        vTemp2 = values[int(randomCouple[1,i])]

        if(v1<vTemp1):
            max1 = randomCouple[0,i]
        if(v2 < vTemp2):
            max2 = randomCouple[1,i]

    return max1,max2


# findMaxCouple will accept a 4x1 array and will return a 2x1 array. The best couple
def findMaxCouple(randomCouple):
    max1 = randomCouple[0]
    max2 = randomCouple[1]
    #print("randomCouple ", randomCouple)
    maxpos1 = 0
    maxpos2 = 1
    #print("values ", randomCouple)
    for i in range(2, 4):
        v1 = randomCouple[i]

        if(v1 > max1):
            if max1 > max2:
                max2 = max1
                maxpos2 = maxpos1
            max1 = v1
            maxpos1 = i
        elif(v1 > max2):
            max2 = v1
            maxpos2 = i

    #print("max 1,2 resp", max1, max2)
    return maxpos1,maxpos2

# coupleSelection will return 2 people (1x2 array) as a couple
def coupleSelection(popSize, ways):
    
    couple = np.zeros((2, ways))
    temp = np.zeros((1, ways))
    winners = np.zeros((1,2))
    couple[0, :] = random.sample(range(0, popSize), ways)
    i = 0
    
    while i < ways:
        x = random.randrange(0,popSize)
        if x not in couple[0, :]:
            temp[0,i] = x
            i += 1

    couple[1,:] = temp
    winners = findMaxParent(couple,ways)
    
    return winners



def tournamentSelection(ways, popSize):
    global values, numberOfCouples

    numberOfCouples = random.randrange(1, int(popSize/2) + 1) # here we decide how many couples there will be
    winners = np.zeros((numberOfCouples, 2)) # this array will store the 2 people to cross

    for i in range(0, numberOfCouples):
        winners[i] = coupleSelection(popSize, ways)
    
    return winners


#crossover function will get 2 people and randomly crossover them (randomly exchange 2,3 or 4 parts)
#pc = probability of crossover
def crossover(couple, genes):
    global population,kids
    parts = random.randrange(2, 5) # this variable will select in how many parts to break each couple
    take = int(genes / parts)
    took = 0
    i = 0 # i will show if take from male or female
    dif = genes - took

    kids = np.zeros((2, genes))
    male = population[int(couple[0])].copy()
    female = population[int(couple[1])].copy()
    #print(male,female)

    while dif >= take:
        if i==0: 
            kids[0,took:(took + take)] = female[took:(took + take)]
            kids[1,took:(took + take)] = male[took:(took + take)]
            i = 1

        elif i==1: 
            kids[1,took:(took + take)] = female[took:(took + take)]
            kids[0,took:(took + take)] = male[took:(took + take)]
            i = 0
        took +=take
        dif = genes - took

    if i==0:
        kids[0,took:(took + dif)] = female[took:(took + dif)]
        kids[1,took:(took + dif)] = male[took:(took + dif)]

    elif i==1:
        kids[1,took:(took + dif)] = female[took:(took + dif)]
        kids[0,took:(took + dif)] = male[took:(took + dif)]


    #print("the kids are",kids,"\n")
    return kids

def mutation(kids, genes):
    muteKind = random.randrange(1, 3) # will choose the kind of mutation
    '''
    1: choose 1 random bit and flip it -> flip mutation
    2: choose 2 random bits and swap them -> swap mutation
    3: choose 3 bits and invert them -> inversion mutation
    '''
    if muteKind == 1:
        #print("mutation:1")
        bit1 = random.randrange(0,genes) # for the first kid
        bit2 = random.randrange(0,genes) # for the second kid
        #print("bit1 = ", bit1)
        #print("bit2 = ", bit2)

        if kids[0,int(bit1)] == 0:
            kids[0,int(bit1)] = 1
        else:
            kids[0,int(bit1)] = 0

        if kids[1,int(bit2)] == 0:
            kids[1,int(bit2)] = 1
        else:
            kids[1,int(bit2)] = 0

    else:
        bit1 = random.sample(range(0, genes), 2)
        bit2 = random.sample(range(0, genes), 2)

        temp1 = kids[0,int(bit1[0])]
        kids[0,int(bit1[0])] = kids[0,int(bit1[1])]
        kids[0,int(bit1[1])] = temp1

        temp2 = kids[1,int(bit2[0])]
        kids[1,int(bit2[0])] = kids[1,int(bit2[1])]
        kids[1,int(bit2[1])] = temp2

    return kids

# this function compares parents and kids and returns the 2 with the highest value
def survivorSelection(kids, parents):
    global genes,values,population
    
    parentValues = np.zeros((2,1))
    kidValues = np.zeros((2,1))

    #a sequence is all the bits (genes) of a chromosome
    survivorSequence = np.zeros((2,genes))
    parentSeq = np.zeros((2,genes))
    
    parentSeq[0,:] = population[int(parents[0])]
    parentSeq[1,:] = population[int(parents[1])]
    bothSequence = np.concatenate((kids, parentSeq), axis=0)

    kidValues = evaluation(kids,2)

    parentValues[0,0] = values[int(parents[0])]
    parentValues[1,0] = values[int(parents[1])]
    
    bothValues = np.concatenate((kidValues, parentValues), axis=0)

    survivors = findMaxCouple(bothValues)
   
    values[int(parents[0])] = bothValues[int(survivors[0])]
    values[int(parents[1])] = bothValues[int(survivors[1])]
    
    #here the survivors are added to the population
    population[int(parents[0])] = bothSequence[int(survivors[0])]
    population[int(parents[1])] = bothSequence[int(survivors[1])]


def population_Crossover_Mutation_Selection(pc,pm):
    global couples,genes, numberOfCouples
    for x in range(0,numberOfCouples):
        okC = random.randrange(1, 11) / 10 # ok for crossover
        if okC <= pc:
            kids = crossover(couples[x,:], genes)
            okM = random.randrange(1, 11) / 10 # ok for mutation
            if okM <= pm:
                #print(" mutation ok")
                kids = mutation(kids,genes)

            survivorSelection(kids, couples[x,:])
            

# now lets ask the user some questions to define problem's characteristics
want_sum = input("what's the summation? ")
want_sum = int(want_sum)
want_prod = input("what's the product? ")
want_prod = int(want_prod)

size = input("whats the size of the initial population? ")
size = int(size)
genes = input("how many genes in each chromosome? ")
genes = int(genes)
ask = input("do you have specific range of number of ones and zeros? y or n? ")
if ask == 'y':
    onesMax = input("maximum number of ones? ")
    onesMin = input("minimum number of ones? ")
else:
    onesMax = 0
    onesMin = 0

pc = input("whats the cross probability:pc? ")
pm = input("whats the mutation probability:pm? ")
pc = float(pc)
pm = float(pm)
generations = input("how many generations? ")
generations = int(generations)

population = np.zeros((size, genes))
values = np.zeros((int(size), 1))
numberOfCouples = 0

# first we create a random population
population = random_B_initialization(population, genes, size, int(onesMin), int(onesMax))

# now we evaluate each person
#values = evaluation(population)
values = evaluation(population, size)
#print(values)

maxVal = np.zeros((generations,1)) # here will the best chromosome's value of each generation will be stored
avgV = np.zeros((generations,1))

# now let's select some people for crossover
print("now the algorithm will implement k - ways tournament selection\n")
k = input("please select the k - ways\n ")
k = int(k)
x = 0
b = 0
index = 0

while (x < generations) and (maxVal[b] <= 0.9):
    couples = tournamentSelection(k, int(size))
    population_Crossover_Mutation_Selection(pc, pm)
    index, maxVal[x] = max(enumerate(values), key=operator.itemgetter(1))
    maxVal[x] = np.amax(values, axis=0)
    #print("max value for ", x, " generation is ", maxVal[x], "\n")
    avgV[x] = avgValue(size)
    
    x += 1
    b = x - 1

print("the best string is: ", population[index])
print("last values are ",values)
print("last population is ",population)
print("the best solution found was ", evaluation(population[index], 1))


plt.figure(1)
plt.plot(avgV, 'ro')
plt.title("Average Generations' Value")
plt.xlabel('Generations')
plt.ylabel('Values')

plt.show()
