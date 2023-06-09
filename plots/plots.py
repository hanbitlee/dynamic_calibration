import numpy as np
import matplotlib.pyplot as plt
import csv
import os

class plotsCalibrationFramework():

    def __init__(self, hyperParameters):
        self.hyperParameters = hyperParameters
        self.candidates = self.hyperParameters.numCandidate
        self.timesteps = self.hyperParameters.numTimeStep
        self.timeVector = np.arange(self.timesteps)

    def plotTSNE(self, compressedRepresentation, agentClusters, clusteringMode, numberOfClusters, strFileName, figureType):
        plt.figure()
        if clusteringMode == 'GMM':
            clusters = [i for i in range(numberOfClusters)]
        elif clusteringMode == 'DPMM':
            clusters = []
            for agent in range(len(agentClusters)):
                if agentClusters[agent] in clusters:
                    pass
                else:
                    clusters.append(agentClusters[agent])
            clusters.sort()
        colors = ['r', 'g', 'b', 'y', 'k', 'coral', 'magenta', 'pink', 'darkcyan',
                  'gold', 'saddlebrown', 'dimgrey', 'crimson', 'mediumpurple', 'rosybrown']
        for agent in range(len(agentClusters)):
            plt.plot(compressedRepresentation[agent][0], compressedRepresentation[agent][1], color=colors[agentClusters[agent] % len(colors)], marker='o', linewidth=0)

        plt.tight_layout()
        if figureType == 2:
            plt.savefig(strFileName, bbox_inches='tight')
        else:
            plt.savefig(strFileName, bbox_inches='tight', dpi=300)
        plt.close()

    def plotTrajectory(self, itr, simResult, simResultCov, observation):
        totalMAPE = 0
        summaryStatistics = simResult.shape[2]
        for ss in range(summaryStatistics):
            plt.close()
            for candidate in range(self.candidates):
                plt.plot(self.timeVector, simResult[candidate][:,ss], label='simulation ')
                plt.fill_between(self.timeVector, np.subtract(simResult[candidate][:,ss],np.sqrt(simResultCov[candidate][:,ss,ss])),
                                 np.add(simResult[candidate][:,ss],np.sqrt(simResultCov[candidate][:,ss,ss])),alpha=0.3)

            plt.plot(self.timeVector, observation[:,ss], label='observation')
            if ss==0:
                MAPE0 = 0
                error = 0
                for time in range(self.hyperParameters.numTimeStep):
                    print(simResult)
                    print(np.shape(simResult))
                    tempSim = simResult[candidate][time][ss]
                    tempVal = observation[time][ss]
                    print(np.shape(tempVal))
                    error += 100*abs(tempSim-tempVal)/tempVal
                MAPE0 = error/self.hyperParameters.numTimeStep
                totalMAPE += MAPE0
                if self.hyperParameters.numOutputDim ==4:
                    plt.title('MAPE for log real GDP at calibration '+str(itr) + ': ' + str(MAPE0) + ('%'))
                elif self.hyperParameters.numOutputDim ==1:
                    if self.hyperParameters.sumStat ==0:
                        plt.title('MAPE for log real GDP at calibration ' + str(itr) + ': ' + str(MAPE0) + ('%'))
                    elif self.hyperParameters.sumStat ==1:
                        plt.title('MAPE for investment at calibration '+str(itr) + ': ' + str(MAPE0) + ('%'))
                    elif self.hyperParameters.sumStat ==2:
                        plt.title('MAPE for wage at calibration '+str(itr) + ': ' + str(MAPE0) + ('%'))
                    elif self.hyperParameters.sumStat ==3:
                        plt.title('MAPE for emloyment at calibration '+str(itr) + ': ' + str(MAPE0) + ('%'))

            elif ss==1:
                MAPE1 = 0
                error = 0
                for time in range(self.hyperParameters.numTimeStep):
                    tempSim = simResult[candidate][time][ss]
                    tempVal = observation[time][ss]
                    error += 100 * abs(tempSim - tempVal) / tempVal
                MAPE1 = error / self.hyperParameters.numTimeStep
                totalMAPE += MAPE1
                plt.title('MAPE for investment at calibration '+str(itr) + ': ' + str(MAPE1) + ('%'))
            elif ss==2:
                MAPE2 = 0
                error = 0
                for time in range(self.hyperParameters.numTimeStep):
                    tempSim = simResult[candidate][time][ss]
                    tempVal = observation[time][ss]
                    error += 100 * abs(tempSim - tempVal) / tempVal
                MAPE2 = error / self.hyperParameters.numTimeStep
                totalMAPE += MAPE2
                plt.title('MAPE for wage at calibration ' + str(itr) + ': ' + str(MAPE2) + ('%'))
            else:
                MAPE3 = 0
                error = 0
                for time in range(self.hyperParameters.numTimeStep):
                    tempSim = simResult[candidate][time][ss]
                    tempVal = observation[time][ss]
                    error += 100 * abs(tempSim - tempVal) / tempVal
                MAPE3 = error / self.hyperParameters.numTimeStep
                totalMAPE += MAPE3
                plt.title('MAPE for employment rate at calibration ' + str(itr) + ': ' + str(MAPE3) + ('%'))
            plt.legend()
            plt.savefig(self.hyperParameters.dir + "SimulationResults_summary_statistics_"+str(ss)
                        +"_iteration_"+str(itr)+".png")

        if self.hyperParameters.numOutputDim ==4:
            MMAPE = totalMAPE/4
            if not os.path.exists('MAPE.csv'):
                with open('MAPE.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([itr, MAPE0])
                    writer.writerow([itr, MAPE1])
                    writer.writerow([itr, MAPE2])
                    writer.writerow([itr, MAPE3])
                    writer.writerow([itr, MMAPE])
                    f.close()
            else:
                with open('MAPE.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([itr, MAPE0])
                    writer.writerow([itr, MAPE1])
                    writer.writerow([itr, MAPE2])
                    writer.writerow([itr, MAPE3])
                    writer.writerow([itr, MMAPE])
                    f.close()
        elif self.hyperParameters.numOutputDim ==1:
            if not os.path.exists('MAPE.csv'):
                with open('MAPE.csv', 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([itr, MAPE0])
                    f.close()
            else:
                with open('MAPE.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([itr, MAPE0])
                    f.close()

    def plotDynamicParameters(self, itr, dicParams):
        dynamicParameters = dicParams['dynamicParameter']

        for estParam in self.hyperParameters.dynParamList:
            plt.close()
            for candidate in range(self.candidates):
                plt.plot(self.timeVector, dynamicParameters[estParam][candidate], label='candidate '+str(candidate))
            plt.legend()
            plt.savefig(self.hyperParameters.dir + "Parameter_"+str(estParam)+"_iteration_" + str(itr) + ".png")

    def plotRegimeDetection(self, itr, regimePerCandidate, mergedRegime):
        yticks = ['Third'+'\n'+'Particle', 'Second'+'\n'+'Particle', 'First'+'\n'+'Particle','Merged'+'\n'+'Regime']
        people = []
        for l in range(len(regimePerCandidate)):
            people.append((str(l) + ' candidate',))
        change_regime = []
        for l in range(len(regimePerCandidate)):
            change_regime.append([0])
            for t in range(self.timesteps - 1):
                if regimePerCandidate[l][t] != regimePerCandidate[l][t + 1]:
                    change_regime[l].append(t + 1)
            if change_regime[l][-1] != self.timesteps:
                change_regime[l].append(self.timesteps)
        # print("change regime : ",change_regime)
        # multi-dimensional data
        data = []
        max_ = 0
        for r in range(len(regimePerCandidate)):
            data.append([])
            if len(change_regime[r]) > max_:
                max_ = len(change_regime[r]) - 1
            for i in range(max_):
                data_tempo = []
                for l in range(len(regimePerCandidate)):
                    if i in range(len(change_regime[r]) - 1):
                        data_tempo.append(change_regime[r][i + 1] - change_regime[r][i])
                    else:
                        data_tempo.append(0)
                data[r].append(data_tempo)
        colors = ['r', 'g', 'b']
        y_positions = []
        plt.close()
        plt.figure(figsize=(10, 5))
        for l in range(len(regimePerCandidate)):
            left = 0
            y_pos = [float(l+1) / float(self.candidates)]
            for t in range(self.timesteps):
                plt.barh(y_pos, width=1, color=colors[int(regimePerCandidate[self.hyperParameters.numCandidate-l-1][t])],
                         align='center', left=left, height=1. / float(self.candidates))
                left += 1
            y_positions += y_pos
        left = 0
        #colorss = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(mergedRegime))]
        colorss = ['black', 'purple', 'yellow', 'cyan', 'magenta', 'lime', 'coral', 'maroon']
        y_pos = [0/float(self.candidates)]
        for t in range(self.timesteps):
            for k in range(len(mergedRegime)):
                if t in mergedRegime[k]:
                    idx = k
            plt.barh(y_pos, width=1, color=colorss[idx % len(colorss)], align='center', left=left, height=1. / float(self.candidates))
            left += 1
        y_positions += y_pos
        plt.xlabel("Time")
        plt.yticks(y_positions, yticks)
        plt.title('Regime Detection Result for each Candidates')
        plt.tight_layout()
        plt.savefig(self.hyperParameters.dir + str(itr) + "-th_Merged_Regime.png")
