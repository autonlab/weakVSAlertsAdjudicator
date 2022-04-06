from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from utils import matches

#CONSTANTS
ABSTAIN = -1
ARTIFACT = 0
REAL = 1

@labeling_function()
def respNK1(x):
    if x.med_rr == -1 or x.respNK1 == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.respNK1,x.med_rr,.15) else ARTIFACT

@labeling_function()
def respNK2(x):
    if x.med_rr == -1 or x.respNK2 == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.respNK2,x.med_rr,.15) else ARTIFACT

@labeling_function()
def respINT(x):
    if x.med_rr == -1 or x.respINT == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.respINT,x.med_rr,.15) else ARTIFACT

@labeling_function()
def respFFT(x):
    if x.med_rr == -1 or x.respFFT == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.respFFT,x.med_rr,.15) else ARTIFACT

@labeling_function()
def respAMP(x):
    if x.respHeight == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return ARTIFACT if x.respHeight < .4 else ABSTAIN

@labeling_function()
def plethFFT(x):
    if x.med_rr == -1 or x.plethFFT == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.plethFFT,x.med_rr,.15) else ABSTAIN

@labeling_function()
def plethNK1_1(x):
    if x.med_rr == -1 or x.plethNK1_1 == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.plethNK1_1,x.med_rr,.15) else ABSTAIN

@labeling_function()
def plethNK1_2(x):
    if x.med_rr == -1 or x.plethNK1_2 == -1:
        return ABSTAIN
    else: #Experiment with switching places of medrr and respnk1
        return REAL if matches(x.plethNK1_2,x.med_rr,.15) else ABSTAIN

lfs = [respNK1,respNK2,respINT,respFFT,respAMP,plethFFT,plethNK1_1,plethNK1_2]