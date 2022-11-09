from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from utils import matches

#CONSTANTS
ABSTAIN = -1
ARTIFACT = 0
REAL = 1


######## Pleth LFs #############
@labeling_function()
def plethNK1(x):
    if x.med_hr == -1 or x.plethNK1 == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethNK1,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethNK2(x):
    if x.med_hr == -1 or x.plethNK2 == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethNK2,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethINT(x):
    if x.med_hr == -1 or x.plethINT == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethINT,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethFFT(x):
    if x.med_hr == -1 or x.plethFFT == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethFFT,x.med_hr,.15) else ARTIFACT

######## Pleth T LFs #############
@labeling_function()
def plethTNK1(x):
    if x.med_hr == -1 or x.plethTNK1 == -1:
        return ABSTAIN
    else:
        return ABSTAIN if matches(x.plethTNK1,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethTNK2(x):
    if x.med_hr == -1 or x.plethTNK2 == -1:
        return ABSTAIN
    else: 
        return ABSTAIN if matches(x.plethTNK2,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethTINT(x):
    if x.med_hr == -1 or x.plethTINT == -1:
        return ABSTAIN
    else: 
        return ABSTAIN if matches(x.plethTINT,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethTFFT(x):
    if x.med_hr == -1 or x.plethTFFT == -1 or x.telemetric == 0:
        return ABSTAIN
    else:
        return ABSTAIN if matches(x.plethTFFT,x.med_hr,.15) else ARTIFACT

######## Pleth matches ECG lead II LFs #############
@labeling_function()
def plethmatchecgNK1(x):
    if x.ecgiiNK1 == -1 or x.plethNK1 == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethNK1,x.ecgiiNK1,.15) else ARTIFACT

@labeling_function()
def plethmatchecgNK2(x):
    if x.ecgiiNK2 == -1 or x.plethNK2 == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethNK2,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethmatchecgINT(x):
    if x.ecgiiINT == -1 or x.plethINT == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethINT,x.med_hr,.15) else ARTIFACT

@labeling_function()
def plethmatchecgFFT(x):
    if x.ecgiiFFT == -1 or x.plethFFT == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethFFT,x.ecgiiFFT,.15) else ARTIFACT

######## Pleth matches ECG lead III LFs #############
@labeling_function()
def plethmatchecg3NK1(x):
    if x.ecgiiiNK1 == -1 or x.plethNK1 == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethNK1,x.ecgiiiNK1,.15) else ARTIFACT

@labeling_function()
def plethmatchecg3NK2(x):
    if x.ecgiiiNK2 == -1 or x.plethNK2 == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethNK2,x.ecgiiiNK2,.15) else ARTIFACT

@labeling_function()
def plethmatchecg3INT(x):
    if x.ecgiiiINT == -1 or x.plethINT == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethINT,x.ecgiiiINT,.15) else ARTIFACT

@labeling_function()
def plethmatchecg3FFT(x):
    if x.ecgiiiFFT == -1 or x.plethFFT == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethFFT,x.ecgiiiFFT,.15) else ARTIFACT

######## Pleth T matches ECG lead II LFs #############
@labeling_function()
def plethTmatchecgNK1(x):
    if x.ecgiiNK1 == -1 or x.plethTNK1 == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethTNK1,x.ecgiiNK1,.15) else ABSTAIN

@labeling_function()
def plethTmatchecgNK2(x):
    if x.ecgiiNK2 == -1 or x.plethTNK2 == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethTNK2,x.med_hr,.15) else ABSTAIN

@labeling_function()
def plethTmatchecgINT(x):
    if x.ecgiiINT == -1 or x.plethTINT == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethTINT,x.med_hr,.15) else ABSTAIN

@labeling_function()
def plethTmatchecgFFT(x):
    if x.ecgiiFFT == -1 or x.plethTFFT == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethTFFT,x.ecgiiFFT,.15) else ABSTAIN

######## Pleth T matches ECG lead III LFs #############
@labeling_function()
def plethTmatchecg3NK1(x):
    if x.ecgiiiNK1 == -1 or x.plethTNK1 == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethTNK1,x.ecgiiiNK1,.15) else ABSTAIN

@labeling_function()
def plethTmatchecg3NK2(x):
    if x.ecgiiiNK2 == -1 or x.plethTNK2 == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethTNK2,x.ecgiiiNK2,.15) else ABSTAIN

@labeling_function()
def plethTmatchecg3INT(x):
    if x.ecgiiiINT == -1 or x.plethTINT == -1:
        return ABSTAIN
    else: 
        return REAL if matches(x.plethTINT,x.ecgiiiINT,.15) else ABSTAIN

@labeling_function()
def plethTmatchecg3FFT(x):
    if x.ecgiiiFFT == -1 or x.plethTFFT == -1:
        return ABSTAIN
    else:
        return REAL if matches(x.plethTFFT,x.ecgiiiFFT,.15) else ABSTAIN

########## Miscellaneous ###########
@labeling_function()
def pulsatilityT(x):
    if x.plethTHeight == -1:
        return ABSTAIN
    else:
        return ARTIFACT if x.plethTHeight < .25 else ABSTAIN

@labeling_function()
def pulsatility(x):
    if x.plethHeight == -1:
        return ABSTAIN
    else:
        return ARTIFACT if x.plethHeight < .25 else ABSTAIN

@labeling_function()
def tachypnea(x):
    if x.med_rr == -1:
        return ABSTAIN
    else:
        return REAL if x.med_rr >20 else ABSTAIN

@labeling_function()
def skew(x):
    if x.skew_rr == 0:
        return ABSTAIN
    else:
        return REAL if x.skew_rr < 0 else ABSTAIN

lfs = [plethNK1, plethNK2, plethINT, plethFFT, pulsatility, pulsatilityT, 
        plethmatchecg3NK1, plethmatchecg3NK2, plethmatchecg3INT, plethmatchecg3FFT, tachypnea,]