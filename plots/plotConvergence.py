import matplotlib.pyplot as plt
import numpy as np
import csv

itr = np.arange(0, 150)
GDP = []
Inv = []
Wage = []
Emp = []
Mean = []

with open('MAPE.csv', 'r', encoding='UTF8', newline='') as f:
    plots = csv.reader(f, delimiter=',')
    for row in plots:
        if row % 5 == 0:
            GDP.append(int(row[1]))
        if row % 5 == 1:
            Inv.append(int(row[1]))
        if row % 5 == 2:
            Wage.append(int(row[1]))
        if row % 5 == 3:
            Emp.append(int(row[1]))
        if row % 5 == 3:
            Mean.append(int(row[1]))

plt.title("Mean Aboslute Percentage Error of Summary Statistics")
plt.plot(itr, GDP, label='realGDP', color='b')
plt.plot(itr, Inv, label='Investment', color='g')
plt.plot(itr, Wage, label='Wage', color='r')
plt.plot(itr, Emp, label='Employment', color='c')
plt.legend()
plt.savefig("MAPE separate.png")
plt.close()

plt.title("Mean Aboslute Percentage Error of Summary Statistics")
plt.plot(itr, Mean, label='Mean', color='b')
plt.legend()
plt.savefig("MAPE all.png")
plt.close()
