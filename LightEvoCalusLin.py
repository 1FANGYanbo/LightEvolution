from LightEvoFuncLin import *
from datetime import datetime
from pathlib import Path
import pandas as pd


tCalcStart = datetime.now()
EAll = [Ei / np.linalg.norm(Ei, 2)]
for q in range(Q):
    zq = q * dz
    ECurr = EAll[q]
    ENext = OneStep(ECurr, zq + dz / 2, dz)
    EAll.append(ENext / np.linalg.norm(ENext, 2))
tCalcEnd = datetime.now()
print("calc time: ", tCalcEnd - tCalcStart)

outDir = "./LightEvolutionLin/Q" + str(Q) + "zTot" + str(zTot) + "/"
pltOut = outDir + "out/"
tValsAll = [q * dz for q in range(0, Q + 1)]

Path(outDir).mkdir(exist_ok=True, parents=True)
outDf = pd.DataFrame(data=EAll)
outDf.to_csv(outDir + "EAllQ" + str(Q) + "data.csv", header=False, index=False)



