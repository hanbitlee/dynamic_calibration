from SimulationModels.simulator import simulation

import os
import numpy as np
import pandas as pd
import csv

def getValidation(hyperParameters, dicParams):
    print("get validation data...")
    if hyperParameters.modelName == 'RealEstateMarketABM':
        file = open(os.getcwd() + '/SimulationModels/'+str(hyperParameters.modelName)+'/Validation/real_validation_data.csv','r')
        lines = file.readlines()
        file.close()
        validation = []
        for line in lines:
            validation.append(line.split(','))
        validation.pop(1)
        validation = np.transpose(validation)
        trueResult = np.zeros((hyperParameters.numTimeStep, hyperParameters.numOutputDim))
        for summaryStatistics in range(len(hyperParameters.outputDim)):
            for time in range(hyperParameters.numTimeStep):
                trueResult[time][summaryStatistics] = float(validation[2+hyperParameters.outputDim[summaryStatistics]][1+time])
        for summaryStatistics in range(len(hyperParameters.outputDim)):
            for time in range(hyperParameters.numTimeStep):
                trueResult[time][4+summaryStatistics] = float(validation[14 + hyperParameters.outputDim[summaryStatistics]][1+time])

    elif hyperParameters.modelName == 'WealthDistributionABM':
        simulator = simulation(hyperParameters, [])
        trueResult, _ = simulator.runParallelSimulation(-1, dicParams)
        trueResult = np.mean(trueResult, 0)

    elif hyperParameters.modelName == 'MacroEconABM':

        filename = os.getcwd() + '/SimulationModels/MacroEconABM/Validation/real_validation_data.csv'

        with open(filename, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                elif line_count == 1:
                    rawData = np.array(row)
                    line_count += 1
                else:
                    rawData = np.vstack([rawData, row])
                    line_count += 1
        rawData = np.transpose(rawData)

        realGDP = calcMovingAvg([float(x) for x in rawData[4]][60:120],4)
        investment = calcMovingAvg([float(x) for x in rawData[7]][60:120],4)
        unemploymentRate = [float(x) for x in rawData[8]][60:120]
        hiringRate = [float(x) for x in rawData[9]][60:120]
        wage = calcMovingAvg([float(x) for x in rawData[10]][60:120],4)

        logRealGDP = [np.math.log(x) for x in realGDP]
        realEmployment = [(100 - x) / 100 for x in unemploymentRate]
        realHiring = [x / 100 for x in hiringRate]
        realEmployHiring = np.mean(np.array([realEmployment, realHiring]), axis=0)

        '''
        #YK_modify_230105
        '''

        if hyperParameters.numOutputDim ==4:
            trueResult = np.hstack([np.expand_dims(x, -1) for x in(logRealGDP, investment, wage, realEmployHiring)])

        elif hyperParameters.numOutputDim ==1:
            if hyperParameters.sumStat == 0:
                trueResult = np.expand_dims(logRealGDP, -1)
            elif hyperParameters.sumStat == 1:
                trueResult = np.expand_dims(investment, -1)
            elif hyperParameters.sumStat == 2:
                trueResult = np.expand_dims(wage, -1)
            elif hyperParameters.sumStat == 3:
                trueResult = np.expand_dims(realEmployHiring, -1)

    return np.array(trueResult)

def calcMovingAvg(lstInput, windowSize):
    lstOutput = []
    for i in range (len(lstInput)):
        if i < windowSize:
            tempCnt = 0
            tempValue = 0
            for j in range(0,i+1):
                tempValue += lstInput[j]
                tempCnt += 1
            lstOutput.append(tempValue/tempCnt)
        else:
            tempCnt = 0
            tempValue = 0
            for j in range(i-windowSize,i+1):
                tempValue += lstInput[j]
                tempCnt += 1
            lstOutput.append(tempValue / tempCnt)

    return lstOutput



