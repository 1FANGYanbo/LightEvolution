import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

inDir = "./LightEvolutionLin/Q1000" + "zTot150" + "/"
inFileName = inDir + "EAllQ1000" + "data.csv"
tReadCsvStart = datetime.now()
inData = pd.read_csv(inFileName, header=None)
tReadCsvEnd = datetime.now()
print("reading csv: ", tReadCsvEnd - tReadCsvStart)
nRow, L = inData.shape

C0 = 1

zTot = 150

Q = nRow - 1

dz = zTot / Q

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 2, 1, projection="3d")

nStart = -500

nRange = range(nStart, nStart + L)

sites = list(nRange)


def strVec2ComplexVec(row):
    retVec = []
    row = np.array(row)
    for elem in row:
        retVec.append(complex(elem))
    return retVec


def strVec2ComplexVecAbs(row):
    retVec = []
    row = np.array(row)
    for elem in row:
        retVec.append(np.abs(complex(elem)))
    return retVec


numOfPics = 30
sep = int((Q + 1) / numOfPics)
pltQVals = list(range(0, Q, sep))
pltQVals.append(Q)
t3dStart = datetime.now()
truncation = 300
for q in pltQVals:
    tValsTmp = [dz * q] * L
    ax1.plot(sites[int(abs(nStart)-truncation/2): int(abs(nStart)+truncation/2)], tValsTmp[:truncation], strVec2ComplexVecAbs(inData.iloc[q, ])[int(abs(nStart)-truncation/2): int(abs(nStart)+truncation/2)], color="black")
# , int(Q / 3), int(Q / 2), int(3 * Q / 4)
pltQRedVals = list([0, int(2 * Q / 3), int(6 * Q / 7)])
pltQRedVals.append(Q)

for q in pltQRedVals:
    tValsTmp = [dz * q] * L
    ax1.plot(sites[int(abs(nStart)-truncation/2): int(abs(nStart)+truncation/2)], tValsTmp[:truncation], strVec2ComplexVecAbs(inData.iloc[q, ])[int(abs(nStart)-truncation/2): int(abs(nStart)+truncation/2)], color="red")

t3dEnd = datetime.now()
ftSize = 17
ax1.view_init(60, -150)
ax1.set_xlim((-int(truncation/2), int(truncation/2)))
ax1.set_xlabel("site", fontsize=ftSize, rotation=60, labelpad=20)
ax1.set_ylabel("z", fontsize=ftSize, rotation=-30, labelpad=20)
ax1.set_zlabel("$|E_{n}|$", fontsize=ftSize, labelpad=15)
ax1.set_title("evolution of wavepacket", fontsize=ftSize)
print("3d time: ", t3dEnd - t3dStart)


# calculates sd
def avgPos(EVec):
    """

    :param EVec:
    :return: <x>
    """
    rst = 0
    for j in range(0, len(nRange)):
        rst += (nRange[j]) * np.abs(EVec[j]) ** 2

    rst /= (np.linalg.norm(EVec, 2)) ** 2
    return rst


def avgPos2(EVec):
    """

    :param EVec:
    :return: <x^{2}>
    """
    rst = 0
    for j in range(0, len(nRange)):
        rst += (nRange[j]) ** 2 * np.abs(EVec[j]) ** 2
    rst /= (np.linalg.norm(EVec, 2)) ** 2
    return rst


def sd(psiVec):
    x = avgPos(psiVec)
    x2 = avgPos2(psiVec)
    s = np.sqrt(np.abs(x2 - x ** 2))
    return s


def skewness(psiVec):
    x = avgPos(psiVec)
    x2 = avgPos2(psiVec)
    s = np.sqrt(np.abs(x2 - x ** 2))
    mu3 = 0
    for j in range(0, len(nRange)):
        mu3 += ((nRange[j] - x) / s) ** 3 * np.abs(psiVec[j]) ** 2
    return mu3


tWidthStart = datetime.now()
pltWidthQVals = list(range(0, Q, 200))
pltWidthQVals.append(Q)
widthVals = [sd(strVec2ComplexVec(inData.iloc[q, ])) for q in pltWidthQVals]

tWidthVals = [q * dz for q in pltWidthQVals]
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(tWidthVals, widthVals, color="black")
ax2.set_xlabel("$z$", fontsize=ftSize, labelpad=3)
ax2.set_ylabel("width", fontsize=ftSize)
ax2.set_ylim((min(widthVals) - 0.1, max(widthVals) + 0.1))
ax2yticks = np.linspace(min(widthVals), max(widthVals), 5)
ax2.set_yticks(ax2yticks)
ax2.set_title("variation of width", fontsize=ftSize)

tWidthEnd = datetime.now()
print("width time: ", tWidthEnd - tWidthStart)

tSkewnessStart = datetime.now()
pltSkewnessQVals = list(range(0, Q, 200))
pltSkewnessQVals.append(Q)
tSkewnessVals = [q * dz for q in pltSkewnessQVals]
skewnessVals = [skewness(strVec2ComplexVec(inData.iloc[q, ])) for q in pltSkewnessQVals]
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(tSkewnessVals, skewnessVals, color="black")
ax3.ticklabel_format(axis='y', scilimits=(-1, -1))
ax3.set_ylim((min(skewnessVals) - 1e-2, max(skewnessVals) + 1e-2))
ax3yticks = list(np.linspace(min(skewnessVals), max(skewnessVals), 5))
ax3.set_yticks(ax3yticks)
ax3.set_xlabel("$z$", fontsize=ftSize, labelpad=3)
ax3.set_ylabel("skewness", fontsize=ftSize)
ax3.set_title("variation of skewness", fontsize=ftSize)
tSkewnessEnd = datetime.now()
print("Skewness time: ", tSkewnessEnd - tSkewnessStart)

tOneDriftStart = datetime.now()
qPlot = int(Q / 2)
tPlot = dz * qPlot
pltDriftQVals = list(range(0, Q, 200))
pltDriftQVals.append(Q)
tDriftVals = [dz * q for q in pltDriftQVals]
avgPosVals = [avgPos(strVec2ComplexVec(inData.iloc[q, ])) for q in pltDriftQVals]
driftVals = [elem - avgPosVals[0] for elem in avgPosVals]

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(tDriftVals, driftVals, color="black")
ax4.set_xlabel("$z$", fontsize=ftSize, labelpad=3)
ax4.set_ylabel("drift", fontsize=ftSize)
ax4.set_ylim((-1, max(driftVals) + 1))
ax4ytickes = np.linspace(min(driftVals), max(driftVals), 5)
ax4.set_yticks(ax4ytickes)
ax4.set_title("motion of center of wave-packet", fontsize=ftSize)
tOneDriftEnd = datetime.now()
print("drift time: ", tOneDriftEnd - tOneDriftStart)

fig.suptitle("Linear Light Evolution", fontsize=ftSize)
plt.savefig(inDir + "EAllLinQ" + str(Q) + "C0" + str(C0) + ".png")
plt.close()
