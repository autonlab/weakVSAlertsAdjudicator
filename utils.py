#Import Statements
import numpy as np
import pandas as pd
from pathlib import Path
from pytz import timezone, utc
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import h5py as h5
from scipy.stats import kurtosis,skew
from neurokit2.signal import signal_rate
from neurokit2.signal import signal_timefrequency

import scipy
from scipy.signal import spectrogram
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

import neurokit2 as nk
import scipy.signal as signal


#Misc. Info
eastern = timezone('US/Eastern')
files_dir = Path('/zfsmladi/originals/')

seriesOptions = {
    'hr': '/data/numerics/HR.HR',
    'rr': '/data/numerics/RR.RR',
    'spo2': '/data/numerics/SpO₂.SpO₂',
    'spo2T':'/data/numerics/SpO₂T.SpO₂T',
    'ecgii':'/data/waveforms/II',
    'ecgiii':'/data/waveforms/III',
    'pleth': '/data/waveforms/Pleth',
    'plethT': '/data/waveforms/PlethT',
    'art': '/data/waveforms/ART',
    'resp': 'data/waveforms/Resp'  
}


#CONSTANTS
WINDOW_SIZE = 60
WF_EXIST_LENGTH_PARAM = .5
NUM_EXIST_LENGTH_PARAM = .1

ABSTAIN = -1
ARTIFACT = 0
REAL = 1

#General Helper Functions
def matches(experimental, actual, THRESHOLD):
    """A function to determine whether a calculated numeric is equivalent to the actual numeric within a margin of error """
    if actual > 20:
        backup = 5
    elif actual > 5:
        backup = 3
    else:
        backup = 2
    thresh = np.maximum(backup, actual * THRESHOLD)
    return np.abs(experimental - actual) <= thresh

def find_FS(h5py_wf, timestamp): 
    """Takes h5py wf data and returns Sampling Frequency as long as there are sufficient data points, else returns 0 (NOT 0.0) """
    try:
        zeit = np.array(h5py_wf[(h5py_wf['time'] > timestamp) & (h5py_wf['time'] < timestamp+60)]['time'])
        if len(zeit) < 20:
            return 0
        timeDiff = np.ediff1d(zeit)
        FS = np.round((1 / np.median(timeDiff)) / 62.5) * 62.5
        return FS
    except:
        return 0

def wf_exists(wf, FS,WF_EXIST_LENGTH_PARAM = .5, WINDOW_SIZE = 60): 
    """Returns whether a given waveform exists, and has sufficient data points based on WF_EXIST_LENGTH_PARAM = .5"""  
    #Make sure Waveform is of sufficient length
    if FS == 0:
        return False
    else:
        return False if len(wf) < WF_EXIST_LENGTH_PARAM * WINDOW_SIZE * FS else True    

def num_exists(num, FS =1, NUM_EXIST_LENGTH_PARAM = .1, WINDOW_SIZE = 60):
    """Returns whether a given numeric exists, and has sufficient data points based on NUM_EXIST_LENGTH_PARAM = .1"""
    #Make sure Numeric is of sufficient length
    return False if len(num) < NUM_EXIST_LENGTH_PARAM * WINDOW_SIZE * FS else True

def promHelper(wf, promOption, promScale = .75):
    """Helper function for goodProm (promOption == 0) and find_wh (promOption == 1)"""
    split = np.array_split(wf,10)

    buckets = []
    PROM_SCALE = promScale
    for el in split:
        buckets.append((np.quantile(el, .75) - np.quantile(el, .25)) * PROM_SCALE)
    if promOption == 0: 
        promThresh = np.mean(buckets)
        return promThresh
    elif promOption == 1:
        wh = np.min([np.median(buckets),np.ptp(wf)])
        return wh

def goodProm(wf,promScale = .75):
    """Returns a prominence value that can be used for SciPy's peak detection method"""
    promThresh = promHelper(wf, 0, promScale)
    return promThresh

def find_wh(wf,altPromScale = 3.5):
    """Finds the relative waveheight of a waveform"""
    wh = promHelper(wf, 1, altPromScale)
    return wh

#Waveform processing helper functions

def second_harmonic(yi,FS,plot = False):
    """Helper function for finding second harmonic of art and pleth waveforms. Uses the waveform created by the peaks of the given waveform"""
    yiyi = nk.rsp_clean(yi,FS)
    
    #peaknk method
    iqrprom = np.quantile(yiyi,.75) - np.quantile(yiyi,.25)
    peaksNK,_ = signal.find_peaks(yiyi,distance = 75,prominence = .4*iqrprom)

    calcnkRR = np.median(signal_rate(peaksNK,FS))
    
    
    xf, yf = signal.periodogram(yiyi, FS, 'bohman', scaling='spectrum')
    peaks,_ = scipy.signal.find_peaks(yf, height = np.max(yf)*.1) #revert to .1 if need be
    
    yPeaks = np.abs(yf[peaks])
    subIndex = np.argmax(yPeaks)
    maxIndex = peaks[subIndex]
    
    potentialRR = xf[peaks]*60
    
    calcRR = xf[maxIndex] * 60
    
    if plot:
        
        fig2, (ax3, ax4, ax5) = plt.subplots(3, 1)
        
        # Interpolated Graph
        ax3.plot(yiyi, 'g')
        ax3.set_title('Cleaned Interpolated Graph ')

        # Periodogram
        ax4.plot(xf[peaks], yf[peaks],"ob")
        ax4.plot(xf, yf)
        ax4.plot(np.ones(len(xf)) * np.mean(yf[peaks]))
        ax4.set_title('Periodogram - Frequency of Respiration')
        ax4.set_xlabel('frequency [Hz]')
        ax4.set_ylabel('Power Spectrum [V**2]')
        ax4.set_xlim([0,10])
#         ax4.set_grid()
        
        #NK peak counting
        ax5.plot(yiyi, 'r')
        ax5.plot(peaksNK,yiyi[peaksNK],'ok')
        ax5.set_title('Peak-Peak Reciprocal Method')
    
        fig2.tight_layout()
        plt.show()
        
    return [calcRR,calcnkRR]

def art_second_harmonic_top(wf,fs,plot = False):
    """Find the RR (Second Harmonic) from the ART waveform"""
    shouldPlot = plot
    FS = fs
    
    #PromThresh
    mw = np.array_split(wf,10)

    kli = []
    PROM_SCALE = 1
    for l in mw:
        kli.append((np.quantile(l,.75) - np.quantile(l,.25))*PROM_SCALE)
    promThresh = np.mean(kli)
    
    tops,_ = scipy.signal.find_peaks(wf, prominence = promThresh,distance = 30)
    
    topY = wf[tops]
    # def find_outliers(wf):
    q1 = np.quantile(topY,.25)
    q3 = np.quantile(topY,.75)
    iqr = q3 - q1

    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    #inner fence bounds
    inner_fence_lower = q1-inner_fence
    inner_fence_upper = q3+inner_fence
    
    #outer fence bounds
    outer_fence_lower = q1-outer_fence
    outer_fence_upper = q3+outer_fence
    
    mask = np.logical_and(topY > outer_fence_lower, topY < outer_fence_upper)
    topX = tops[mask]
    topY = topY[mask]
    
    lis = []
    for a in [t - s for s, t in zip(topX, topX[1:])]:
        lis.append(a)

    q12 = np.quantile(lis,.25)
    q32 = np.quantile(lis,.75)
    iqr2 = q32 - q12

    inner_fence2 = 1.5*iqr2
    outer_fence2 = 3*iqr2

    #inner fence bounds
    inner_fence_lower2 = q12-inner_fence2
    inner_fence_upper2 = q32+inner_fence2
    
    #outer fence bounds
    outer_fence_lower2 = q12-outer_fence2
    outer_fence_upper2 = q32+outer_fence2
    
    mask2 = (np.logical_or(lis < outer_fence_lower2, lis > outer_fence_upper2))
    
    fencePost = np.arange(len(mask2))[mask2]

    newFencePost = fencePost.tolist()
    newFencePost.append(0)
    newFencePost.append(len(mask2))
    fp = np.sort(np.array(newFencePost))
    lis2 = []
    for a in [t - s for s, t in zip(fp, fp[1:])]:
        lis2.append(a)
    lis2 = np.array(lis2)
    first, second = fp[np.argmax(lis2)],fp[np.argmax(lis2)+1]
    start = (first + 1) if first in fencePost else first
    end = (second) if second in fencePost else (second + 1)
    top = topX[start:end+1] #remove ends of peaks if fencepost
    
    x = top
    y = wf[top]
    xi = np.linspace(x[0], x[-1], x[-1]-x[0]+1)

    # use fitpack2 method
    ius = InterpolatedUnivariateSpline(x, y)
    yi = ius(xi)

    if shouldPlot:
        fig1, (ax1, ax2) = plt.subplots(2, 1)
        fig1.suptitle('Secondary Harmonic Plots')

        # Top Peaks of ART
        ax1.plot(x, y,"ob")
        ax1.plot(wf)
        ax1.set_title('Top Peaks of ART')

        # Peaks plus interpolation
        ax2.plot(x, y, 'bo')
        ax2.plot(xi, yi, 'g')
        ax2.set_title('Interpolation using univariate spline')

        fig1.tight_layout()


    rr = second_harmonic(yi,FS,shouldPlot)

    return (rr)


def plethProcessingSP(wf,FS):
    """Pleth Processing function for SpO2 vital sign alerts"""
    cleaned = nk.ppg_clean(wf,FS)
    
    #Modified Periodogram
    xf, yf = scipy.signal.periodogram(cleaned, FS, 'bohman', scaling='spectrum')
    peaks,_ = scipy.signal.find_peaks(yf,height = np.max(yf)*.25)#height = np.max(yf)*.25
    subIndex = np.argmax(yf[peaks])
    maxIndex = peaks[subIndex]
    ppgFFT = xf[maxIndex]*60
    
    #Other Methods
    
    info = nk.ppg_findpeaks(wf,FS)
    rpeaks = info['PPG_Peaks']
    
    #PromThresh
    promThresh = goodProm(wf,1)
    
    tops,_ = scipy.signal.find_peaks(wf, prominence = promThresh,distance = 30)
    
    if len(tops) > 0:
        if len(rpeaks) > 0:
            ppgINT = np.median(nk.signal_rate(np.intersect1d(tops,rpeaks),FS))
            ppgNK2 = np.median(nk.signal_rate(rpeaks,FS))
        else:
            ppgINT = 0
            ppgNK2 = 0
        ppgNK1 = np.median(nk.signal_rate(tops,FS))
    elif len(tops) == 0:
        ppgINT = 0
        ppgNK1 = 0
        ppgNK2 = 0
    
    return [ppgFFT,ppgINT,ppgNK1,ppgNK2]

def plethProcessingRR(wf,fs,plot = False):
    """Pleth processing function for Respiratory Rate alerts"""
    shouldPlot = plot
    FS = fs
    
    xf, yf = signal.periodogram(wf, FS, 'flattop', scaling='spectrum')
    zf = xf[xf<1]
    yf = yf[xf<1]
    peaks,_ = scipy.signal.find_peaks(yf)#height = np.max(yf)*.25
    subIndex = np.argmax(yf[peaks])
    maxIndex = peaks[subIndex]
    plethDirectFFT = zf[maxIndex]*60
    
    
    info = nk.ppg_findpeaks(wf,FS)
    tops = info['PPG_Peaks']
    
    topY = wf[tops]
    # def find_outliers(wf):
    q1 = np.quantile(topY,.25)
    q3 = np.quantile(topY,.75)
    iqr = q3 - q1

    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    #inner fence bounds
    inner_fence_lower = q1-inner_fence
    inner_fence_upper = q3+inner_fence
    
    #outer fence bounds
    outer_fence_lower = q1-outer_fence
    outer_fence_upper = q3+outer_fence
    
    mask = np.logical_and(topY > outer_fence_lower, topY < outer_fence_upper)
    topX = tops[mask]
    topY = topY[mask]
    
    lis = []
    for a in [t - s for s, t in zip(topX, topX[1:])]:
        lis.append(a)

    q12 = np.quantile(lis,.25)
    q32 = np.quantile(lis,.75)
    iqr2 = q32 - q12

    inner_fence2 = 1.5*iqr2
    outer_fence2 = 3*iqr2

    #inner fence bounds
    inner_fence_lower2 = q12-inner_fence2
    inner_fence_upper2 = q32+inner_fence2
    
    #outer fence bounds
    outer_fence_lower2 = q12-outer_fence2
    outer_fence_upper2 = q32+outer_fence2
    
    mask2 = (np.logical_or(lis < outer_fence_lower2, lis > outer_fence_upper2))
    
    fencePost = np.arange(len(mask2))[mask2]

    newFencePost = fencePost.tolist()
    newFencePost.append(0)
    newFencePost.append(len(mask2))
    fp = np.sort(np.array(newFencePost))
#     print(fp)
    lis2 = []
    for a in [t - s for s, t in zip(fp, fp[1:])]:
        lis2.append(a)
    lis2 = np.array(lis2)
#     print(lis2)
    first, second = fp[np.argmax(lis2)],fp[np.argmax(lis2)+1]
#     print([first,second])
    start = (first + 1) if first in fencePost else first
    end = (second) if second in fencePost else (second + 1)
#     print([start,end])
    top = topX[start:end+1] #remove ends of peaks if fencepost
    
    x = top
    y = wf[top]
    xi = np.linspace(x[0], x[-1], x[-1]-x[0]+1)

    # use fitpack2 method
    ius = InterpolatedUnivariateSpline(x, y)
    yi = ius(xi)

    if shouldPlot:
        fig1, (ax1, ax2) = plt.subplots(2, 1)
        fig1.suptitle('Secondary Harmonic Plots')

        # Top Peaks of ART
        ax1.plot(x, y,"ob")
        ax1.plot(wf)
        ax1.set_title('Top Peaks of ART')

        # Peaks plus interpolation
        ax2.plot(x, y, 'bo')
        ax2.plot(xi, yi, 'g')
        ax2.set_title('Interpolation using univariate spline')

        fig1.tight_layout()


    rr = second_harmonic(yi,FS,shouldPlot)

    return ([rr,plethDirectFFT])

def ecgProcessing(wf,FS):
    """ECG processing function"""
    cleaned = nk.ecg_clean(wf,FS)
    
    #Modified Periodogram
    xf, yf = scipy.signal.periodogram(nk.ecg_clean(wf,FS), FS, 'bohman', scaling='spectrum')
    peaks,_ = scipy.signal.find_peaks(yf,height = np.max(yf)*.25)#height = np.max(yf)*.25
    subIndex = np.argmax(yf[peaks])
    maxIndex = peaks[subIndex]
    ecgFFT = xf[maxIndex]*60
    
    #Other Methods
    tops,_ = scipy.signal.find_peaks(cleaned, prominence = np.max(cleaned) *.6, distance = 60 * FS/250, height = .3)
    _, ecg_rpeaks_from_nk = nk.ecg_peaks(cleaned, sampling_rate = FS, method='neurokit')
    rpeaks = ecg_rpeaks_from_nk['ECG_R_Peaks']
    if len(tops) > 0:
        if len(rpeaks) > 0:
            ecgINT = np.median(nk.signal_rate(np.intersect1d(tops,rpeaks),FS))
            ecgNK2 = np.median(nk.signal_rate(rpeaks,FS))
        else:
            ecgINT = 0
            ecgNK2 = 0
        ecgNK1 = np.median(nk.signal_rate(tops,FS))
    elif len(tops) == 0:
        ecgINT = 0
        ecgNK1 = 0
        ecgNK2 = 0
    
    return [ecgFFT,ecgINT,ecgNK1,ecgNK2]



#Processing Functions

def spProcess(n,df, pathToSaveData):
    """Process all SpO2 alerts to find relevant features"""

    features = np.array(['med_hr', 'std_hr', 'numdp_hr', 'q1_hr', 'q3_hr', 'kurt_hr',
       'skew_hr', 'med_rr', 'std_rr', 'numdp_rr', 'q1_rr', 'q3_rr',
       'kurt_rr', 'skew_rr', 'med_spo2', 'std_spo2', 'numdp_spo2',
       'q1_spo2', 'q3_spo2', 'kurt_spo2', 'skew_spo2', 'med_spo2T',
       'std_spo2T', 'numdp_spo2T', 'q1_spo2T', 'q3_spo2T', 'kurt_spo2T',
       'skew_spo2T', 'med_ecgii', 'std_ecgii', 'numdp_ecgii', 'q1_ecgii',
       'q3_ecgii', 'kurt_ecgii', 'skew_ecgii', 'med_ecgiii', 'std_ecgiii',
       'numdp_ecgiii', 'q1_ecgiii', 'q3_ecgiii', 'kurt_ecgiii',
       'skew_ecgiii', 'med_pleth', 'std_pleth', 'numdp_pleth', 'q1_pleth',
       'q3_pleth', 'kurt_pleth', 'skew_pleth', 'med_plethT', 'std_plethT',
       'numdp_plethT', 'q1_plethT', 'q3_plethT', 'kurt_plethT',
       'skew_plethT', 'med_art', 'std_art', 'numdp_art', 'q1_art',
       'q3_art', 'kurt_art', 'skew_art', 'med_resp', 'std_resp',
       'numdp_resp', 'q1_resp', 'q3_resp', 'kurt_resp', 'skew_resp',
       'ecgiiFFT', 'ecgiiINT', 'ecgiiNK1', 'ecgiiNK2', 'ecgiiiFFT',
       'ecgiiiINT', 'ecgiiiNK1', 'ecgiiiNK2', 'plethFFT', 'plethINT',
       'plethNK1', 'plethNK2', 'plethTFFT', 'plethTINT', 'plethTNK1',
       'plethTNK2', 'artFFT', 'artINT', 'artNK1', 'artNK2', 'tooEarly',
       'plethHeight', 'plethTHeight'])

    ###################
    ##Beginning Stuff##
    ###################
    
    x = ['med_','std_','numdp_','q1_','q3_','kurt_','skew_']
    y = ['hr','rr','spo2','spo2T','ecgii','ecgiii','pleth','plethT','art','resp']
    nums = ['hr','rr','spo2','spo2T']
    wfs = ['ecgii','ecgiii','pleth','plethT','art','resp']

    
    #constants
    WINDOW_SIZE = 60
    WF_EXIST_LENGTH_PARAM = .5
    NUM_EXIST_LENGTH_PARAM = .1
    
    #Find all alerts with fileID n
    subAlert = df.loc[df.fileID == n]
    match_ind = pd.Series(subAlert.index)
    fName = subAlert.iloc[0].filename
    startTimes = np.transpose(np.array([subAlert.start0.values,subAlert.start1.values,subAlert.start2.values]))
    
    # Open the file using the h5py lib
    h5file = h5.File(files_dir / fName, mode='r')
    
    #Basetime
    meta_obj = json.loads(h5file.attrs['.meta'])
    time_obj = datetime.strptime(meta_obj['time_origin'][:-4], "%Y-%m-%d %H:%M:%S.%f")
    time_obj = eastern.localize(time_obj)
    basetime = time_obj.timestamp()
    
    ##############################
    ##Earliest Waveform Time 1/2##
    ##############################
    
    time_min =[]
    
    ##########################################
    ##Load all numerics and waveforms & EWFT##
    ##########################################
    #skeleton for h5py data
    h5py = dict.fromkeys(y)
    #skeleton for exists
    exists = {k: False for k in y}
    #skeleton for times
    timeEarly = dict.fromkeys(wfs)
    
    ### Load All Numerics ###
    for numeric in nums:
        if seriesOptions[numeric] in h5file:
            exists[numeric] = True
            h5py[numeric] = h5file[seriesOptions[numeric]]
 
    ### Load All Waveforms ###
    for waveform in wfs:
        if(seriesOptions[waveform] in h5file):
            exists[waveform] = True       
            h5py[waveform] = h5file[seriesOptions[waveform]]
            
            timeEarly[waveform] = h5py[waveform]['time'][0]
            time_min.append(timeEarly[waveform])

        
    
    ##############################
    ##Earliest Waveform Time 2/2##
    ##############################
    
    EWFT = np.min(time_min) + basetime
    
    
    
    #######################################
    ##Loop through Individual WFs in File##
    #######################################
    
    individualAlerts = []
    for alert in startTimes:
        #Dictionary of Features
        ds = {k: [] for k in features}
        for time in tqdm(alert):
            ######################################
            ##Sampling Frequencies for Waveforms##
            ######################################

            #Skeleton for sampling frequencies
            FS = dict.fromkeys(wfs)

            #FS for Resp
            FS['resp'] = 62.5 if exists['resp'] else 0

            #FS for Pleth
            FS['pleth'] = 125.0 if exists['pleth'] else 0

            #FS for PlethT
            FS['plethT'] = 125.0 if exists['plethT'] else 0

            #FS for ART
            FS['art'] = 125.0 if exists['art'] else 0

            #FS for ECGii
            FS['ecgii'] = find_FS(h5py['ecgii'],time-basetime) if exists['ecgii'] else 0

            #FS for ECGiii
            FS['ecgiii'] = find_FS(h5py['ecgiii'],time-basetime) if exists['ecgiii'] else 0

            #skeleton for numerics and waveform data
            data = dict.fromkeys(y)
            #Too Early of an Alert
            ds['tooEarly'].append(True) if time - EWFT < 300 else ds['tooEarly'].append(False)     
            
            #Looping stuff for Numerics
            for numeric in nums:
                if exists[numeric]:
                    data[numeric] = np.array(h5py[numeric][(h5py[numeric]['time'] > time-basetime) & (h5py[numeric]['time'] < time+60-basetime)]['value'])
                    if num_exists(data[numeric]):
                        ds['med_' + numeric].append(np.median(data[numeric]))
                        ds['std_'+numeric].append(np.std(data[numeric]))
                        ds['numdp_'+numeric].append(len(data[numeric]))
                        ds['q1_'+numeric].append(np.quantile(data[numeric],.25))
                        ds['q3_'+numeric].append(np.quantile(data[numeric],.75))
                        ds['kurt_'+numeric].append(kurtosis(data[numeric]))
                        ds['skew_'+numeric].append(skew(data[numeric]))
                    else:
                        for f in x:
#                             print('bad')#^&*#
                            ds[f+numeric].append(np.nan)     
                else:
                    for f in x:
                        ds[f+numeric].append(np.nan)
            
            #Looping stuff for Waveforms       
            for waveform in wfs:
                if exists[waveform]:
                    data[waveform] = np.array(h5py[waveform][(h5py[waveform]['time'] > time-basetime) & (h5py[waveform]['time'] < time+60-basetime)]['value'])
                    if wf_exists(data[waveform],FS[waveform]):
                        ds['med_' + waveform].append(np.median(data[waveform]))
                        ds['std_'+waveform].append(np.std(data[waveform]))
                        ds['numdp_'+waveform].append(len(data[waveform]))
                        ds['q1_'+waveform].append(np.quantile(data[waveform],.25))
                        ds['q3_'+waveform].append(np.quantile(data[waveform],.75))
                        ds['kurt_'+waveform].append(kurtosis(data[waveform]))
                        ds['skew_'+waveform].append(skew(data[waveform]))
                    else:
                        for f in x:
                            ds[f+waveform].append(np.nan)     
                else:
                    for f in x:
                        ds[f+waveform].append(np.nan)

            ### Manual Features ###

            #ECGii Features:ecgProcessing(wf,FS)
            if exists['ecgii']:
                data['ecgii'] = np.array(h5py['ecgii'][(h5py['ecgii']['time'] > time-basetime) & (h5py['ecgii']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['ecgii'],FS['ecgii']):
                    try:
                        ecgii_result = ecgProcessing(data['ecgii'],FS['ecgii'])
                    except:
                        ecgii_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['ecgiiFFT'].append(ecgii_result[0])
                    ds['ecgiiINT'].append(ecgii_result[1])
                    ds['ecgiiNK1'].append(ecgii_result[2])
                    ds['ecgiiNK2'].append(ecgii_result[3])
                    
                else:
                    ds['ecgiiFFT'].append(np.nan)
                    ds['ecgiiINT'].append(np.nan)
                    ds['ecgiiNK1'].append(np.nan)
                    ds['ecgiiNK2'].append(np.nan)
            
            else:
                ds['ecgiiFFT'].append(np.nan)
                ds['ecgiiINT'].append(np.nan)
                ds['ecgiiNK1'].append(np.nan)
                ds['ecgiiNK2'].append(np.nan)
            
            #ECGiii Features:ecgProcessing(wf,FS)
            if exists['ecgiii']:
                data['ecgiii'] = np.array(h5py['ecgiii'][(h5py['ecgiii']['time'] > time-basetime) & (h5py['ecgiii']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['ecgiii'],FS['ecgiii']):
                    try:
                        ecgiii_result = ecgProcessing(data['ecgiii'],FS['ecgiii'])
                    except:
                        ecgiii_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['ecgiiiFFT'].append(ecgiii_result[0])
                    ds['ecgiiiINT'].append(ecgiii_result[1])
                    ds['ecgiiiNK1'].append(ecgiii_result[2])
                    ds['ecgiiiNK2'].append(ecgiii_result[3])
                    
                else:
                    ds['ecgiiiFFT'].append(np.nan)
                    ds['ecgiiiINT'].append(np.nan)
                    ds['ecgiiiNK1'].append(np.nan)
                    ds['ecgiiiNK2'].append(np.nan)
            
            else:
                ds['ecgiiiFFT'].append(np.nan)
                ds['ecgiiiINT'].append(np.nan)
                ds['ecgiiiNK1'].append(np.nan)
                ds['ecgiiiNK2'].append(np.nan)
       
        #art features:ecgProcessing(wf,FS)
            if exists['art']:
                data['art'] = np.array(h5py['art'][(h5py['art']['time'] > time-basetime) & (h5py['art']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['art'],FS['art']):
                    try:
                        art_result = ecgProcessing(data['art'],FS['art'])
                    except:
                        art_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['artFFT'].append(art_result[0])
                    ds['artINT'].append(art_result[1])
                    ds['artNK1'].append(art_result[2])
                    ds['artNK2'].append(art_result[3])
                    
                else:
                    ds['artFFT'].append(np.nan)
                    ds['artINT'].append(np.nan)
                    ds['artNK1'].append(np.nan)
                    ds['artNK2'].append(np.nan)
            
            else:
                ds['artFFT'].append(np.nan)
                ds['artINT'].append(np.nan)
                ds['artNK1'].append(np.nan)
                ds['artNK2'].append(np.nan)
        
        
        #pleth features:plethProcessing(wf,FS)
            if exists['pleth']:
                data['pleth'] = np.array(h5py['pleth'][(h5py['pleth']['time'] > time-basetime) & (h5py['pleth']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['pleth'],FS['pleth']):
                    pleth = nk.ppg_clean(data['pleth'],FS['pleth'])
                    #pleth Height
                    ds['plethHeight'].append(find_wh(pleth)) 
                    try:
                        pleth_result = plethProcessingSP(data['pleth'],FS['pleth'])
                    except:
                        pleth_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['plethFFT'].append(pleth_result[0])
                    ds['plethINT'].append(pleth_result[1])
                    ds['plethNK1'].append(pleth_result[2])
                    ds['plethNK2'].append(pleth_result[3])
                    
                else:
                    ds['plethHeight'].append(np.nan) 
                    ds['plethFFT'].append(np.nan)
                    ds['plethINT'].append(np.nan)
                    ds['plethNK1'].append(np.nan)
                    ds['plethNK2'].append(np.nan)
            
            else:
                ds['plethHeight'].append(np.nan) 
                ds['plethFFT'].append(np.nan)
                ds['plethINT'].append(np.nan)
                ds['plethNK1'].append(np.nan)
                ds['plethNK2'].append(np.nan)
  
        #plethT features:plethProcessing(wf,FS)
            if exists['plethT']:
                data['plethT'] = np.array(h5py['plethT'][(h5py['plethT']['time'] > time-basetime) & (h5py['plethT']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['plethT'],FS['plethT']):
                    plethT = nk.ppg_clean(data['plethT'],FS['plethT'])
                    #pleth Height
                    ds['plethTHeight'].append(find_wh(plethT)) 
                    try:
                        plethT_result = plethProcessingSP(data['plethT'],FS['plethT'])
                    except:
                        plethT_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['plethTFFT'].append(plethT_result[0])
                    ds['plethTINT'].append(plethT_result[1])
                    ds['plethTNK1'].append(plethT_result[2])
                    ds['plethTNK2'].append(plethT_result[3])
                    
                else:
                    ds['plethTHeight'].append(np.nan) 
                    ds['plethTFFT'].append(np.nan)
                    ds['plethTINT'].append(np.nan)
                    ds['plethTNK1'].append(np.nan)
                    ds['plethTNK2'].append(np.nan)
            
            else:
                ds['plethTHeight'].append(np.nan) 
                ds['plethTFFT'].append(np.nan)
                ds['plethTINT'].append(np.nan)
                ds['plethTNK1'].append(np.nan)
                ds['plethTNK2'].append(np.nan)
        
        #Final Assignment of all variables into respective DF column    
        subResult = pd.DataFrame([ds])        
        individualAlerts.append(subResult)      
    result = pd.concat(individualAlerts)
    
    #change index result df with indices of subAlert
    result.set_index(match_ind,inplace=True)
    result.to_pickle(pathToSaveData + f"/{fName[:-3]}.pkl")          
        
def rrProcess(n,df, pathToSaveData):
    """Fully processing RR alerts ==> featurizing"""
    features = np.array(['med_hr', 'std_hr', 'mean_hr', 'q1_hr', 'q3_hr', 'kurt_hr',
       'skew_hr', 'med_rr', 'std_rr', 'mean_rr', 'q1_rr', 'q3_rr',
       'kurt_rr', 'skew_rr', 'med_spo2', 'std_spo2', 'mean_spo2',
       'q1_spo2', 'q3_spo2', 'kurt_spo2', 'skew_spo2', 'med_ecgii',
       'std_ecgii', 'mean_ecgii', 'q1_ecgii', 'q3_ecgii', 'kurt_ecgii',
       'skew_ecgii', 'med_ecgiii', 'std_ecgiii', 'mean_ecgiii',
       'q1_ecgiii', 'q3_ecgiii', 'kurt_ecgiii', 'skew_ecgiii',
       'med_pleth', 'std_pleth', 'mean_pleth', 'q1_pleth', 'q3_pleth',
       'kurt_pleth', 'skew_pleth', 'med_art', 'std_art', 'mean_art',
       'q1_art', 'q3_art', 'kurt_art', 'skew_art', 'med_resp', 'std_resp',
       'mean_resp', 'q1_resp', 'q3_resp', 'kurt_resp', 'skew_resp',
       'respHeight', 'respFFT', 'respINT', 'respNK1', 'respNK2',
       'plethFFT', 'plethNK1', 'artFFT', 'artNK1', 'ecgiiFFT', 'ecgiiINT',
       'ecgiiNK1', 'ecgiiNK2', 'ecgiiiFFT', 'ecgiiiINT', 'ecgiiiNK1',
       'ecgiiiNK2', 'tooEarly', 'med_spo2T', 'std_spo2T', 'mean_spo2T',
       'q1_spo2T', 'q3_spo2T', 'kurt_spo2T', 'skew_spo2T', 'med_plethT',
       'std_plethT', 'mean_plethT', 'q1_plethT', 'q3_plethT',
       'kurt_plethT', 'skew_plethT', 'plethTFFT', 'plethTNK1'])
    ###################
    ##Beginning Stuff##
    ###################
    
    x = ['med_','std_','mean_','q1_','q3_','kurt_','skew_']
    y = ['hr','rr','spo2','spo2T','ecgii','ecgiii','pleth','plethT','art','resp']
    nums = ['hr','rr','spo2','spo2T']
    wfs = ['ecgii','ecgiii','pleth','plethT','art','resp']

    
    #constants
    WINDOW_SIZE = 60
    WF_EXIST_LENGTH_PARAM = .5
    NUM_EXIST_LENGTH_PARAM = .1
    
    #Find all alerts with fileID n
    subAlert = df.loc[df.fileID == n]
    match_ind = pd.Series(subAlert.index)
    fName = subAlert.iloc[0].filename
    startTimes = np.transpose(np.array([subAlert.start0.values,subAlert.start1.values,subAlert.start2.values]))
    
    # Open the file using the h5py lib
    h5file = h5.File(files_dir / fName, mode='r')
    
    #Basetime
    meta_obj = json.loads(h5file.attrs['.meta'])
    time_obj = datetime.strptime(meta_obj['time_origin'][:-4], "%Y-%m-%d %H:%M:%S.%f")
    time_obj = eastern.localize(time_obj)
    basetime = time_obj.timestamp()
    
    ##############################
    ##Earliest Waveform Time 1/2##
    ##############################
    
    time_min =[]
    
    ##########################################
    ##Load all numerics and waveforms & EWFT##
    ##########################################
    #skeleton for h5py data
    h5py = dict.fromkeys(y)
    #skeleton for exists
    exists = {k: False for k in y}
    #skeleton for times
    timeEarly = dict.fromkeys(wfs)
    
    ### Load All Numerics ###
    for numeric in nums:
        if seriesOptions[numeric] in h5file:
            exists[numeric] = True
            h5py[numeric] = h5file[seriesOptions[numeric]]
 
    ### Load All Waveforms ###
    for waveform in wfs:
        if(seriesOptions[waveform] in h5file):
            exists[waveform] = True       
            h5py[waveform] = h5file[seriesOptions[waveform]]
            
            timeEarly[waveform] = h5py[waveform]['time'][0]
            time_min.append(timeEarly[waveform])

        
    
    ##############################
    ##Earliest Waveform Time 2/2##
    ##############################
    
    EWFT = np.min(time_min) + basetime

    #######################################
    ##Loop through Individual WFs in File##
    #######################################
    
    individualAlerts = []
    for alert in startTimes:
        #Dictionary of Features
        ds = {k: [] for k in features}
        for time in tqdm(alert):
            ######################################
            ##Sampling Frequencies for Waveforms##
            ######################################

            #Skeleton for sampling frequencies
            FS = dict.fromkeys(wfs)

            #FS for Resp
            FS['resp'] = 62.5 if exists['resp'] else 0

            #FS for Pleth
            FS['pleth'] = 125.0 if exists['pleth'] else 0

            #FS for PlethT
            FS['plethT'] = 125.0 if exists['plethT'] else 0

            #FS for ART
            FS['art'] = 125.0 if exists['art'] else 0

            #FS for ECGii
            FS['ecgii'] = find_FS(h5py['ecgii'],time-basetime) if exists['ecgii'] else 0

            #FS for ECGiii
            FS['ecgiii'] = find_FS(h5py['ecgiii'],time-basetime) if exists['ecgiii'] else 0
            
            #skeleton for numerics and waveform data
            data = dict.fromkeys(y)
            #Too Early of an Alert
            ds['tooEarly'].append(True) if time - EWFT < 300 else ds['tooEarly'].append(False)     
            
            #Looping stuff for Numerics
            for numeric in nums:
                if exists[numeric]:
                    data[numeric] = np.array(h5py[numeric][(h5py[numeric]['time'] > time-basetime) & (h5py[numeric]['time'] < time+60-basetime)]['value'])
                    if num_exists(data[numeric]):
                        ds['med_' + numeric].append(np.median(data[numeric]))
                        ds['std_'+numeric].append(np.std(data[numeric]))
                        ds['mean_'+numeric].append(np.mean(data[numeric]))
                        ds['q1_'+numeric].append(np.quantile(data[numeric],.25))
                        ds['q3_'+numeric].append(np.quantile(data[numeric],.75))
                        ds['kurt_'+numeric].append(kurtosis(data[numeric]))
                        ds['skew_'+numeric].append(skew(data[numeric]))
                    else:
                        for f in x:
                            ds[f+numeric].append(np.nan)     
                else:
                    for f in x:
                        ds[f+numeric].append(np.nan)
            
            #Looping stuff for Waveforms       
            for waveform in wfs:
                if exists[waveform]:
                    data[waveform] = np.array(h5py[waveform][(h5py[waveform]['time'] > time-basetime) & (h5py[waveform]['time'] < time+60-basetime)]['value'])
                    if wf_exists(data[waveform],FS[waveform]):
                        ds['med_' + waveform].append(np.median(data[waveform]))
                        ds['std_'+waveform].append(np.std(data[waveform]))
                        ds['mean_'+waveform].append(np.mean(data[waveform]))
                        ds['q1_'+waveform].append(np.quantile(data[waveform],.25))
                        ds['q3_'+waveform].append(np.quantile(data[waveform],.75))
                        ds['kurt_'+waveform].append(kurtosis(data[waveform]))
                        ds['skew_'+waveform].append(skew(data[waveform]))
                    else:
                        for f in x:
                            ds[f+waveform].append(np.nan)     
                else:
                    for f in x:
                        ds[f+waveform].append(np.nan)

            ### Manual Features ###
            
            # Resp Features
            if exists['resp']:
                data['resp'] = np.array(h5py['resp'][(h5py['resp']['time'] > time-basetime) & (h5py['resp']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['resp'],FS['resp']):
                    resp = nk.rsp_clean(data['resp'],FS['resp'])
                    #Resp Height
                    ds['respHeight'].append(find_wh(resp))

                    #Periodogram Method
                    xf, yf = signal.periodogram(resp, FS['resp'], 'bohman', scaling='spectrum')
                    peaks,_ = scipy.signal.find_peaks(yf, height = np.max(yf) * .25)
                    subIndex = np.argmax(yf[peaks])
                    maxIndex = peaks[subIndex]
                    ds['respFFT'].append(xf[maxIndex]*60)

                    #Other Methods
                    tops,_ = scipy.signal.find_peaks(resp, prominence = goodProm(resp,.75),distance = 40)
                    info = nk.rsp_findpeaks(resp, sampling_rate=FS['resp'], method="biosppy")
                    rpeak = info['RSP_Peaks']
                    if len(tops) > 0:
                        if len(rpeak) > 0:
                            ds['respINT'].append(np.median(nk.signal_rate(np.intersect1d(tops,rpeak),FS['resp'])))
                        else:
                            ds['respINT'].append(0)
                        ds['respNK1'].append(np.median(nk.signal_rate(tops,FS['resp'])))
                    elif len(tops) == 0:
                        ds['respINT'].append(0)
                        ds['respNK1'].append(0)

                    signals, info = nk.rsp_process(resp,FS['resp'],method = 'biosppy')
                    if len(signals['RSP_Rate']) > 0:
                        ds['respNK2'].append(np.median(signals['RSP_Rate']))
                    else:
                        ds['respNK2'].append(0)
                else:
                    ds['respFFT'].append(np.nan)
                    ds['respINT'].append(np.nan)
                    ds['respNK1'].append(np.nan)
                    ds['respNK2'].append(np.nan)
                    ds['respHeight'].append(np.nan)
            else:
                ds['respFFT'].append(np.nan)
                ds['respINT'].append(np.nan)
                ds['respNK1'].append(np.nan)
                ds['respNK2'].append(np.nan)
                ds['respHeight'].append(np.nan)
                
            #ART Features: art_second_harmonic_top(wf,fs  
            if exists['art']:
                data['art'] = np.array(h5py['art'][(h5py['art']['time'] > time-basetime) & (h5py['art']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['art'],FS['art']):
                    try:
                        art_result = art_second_harmonic_top(data['art'],FS['art'])
                    except:
                        art_result = [np.nan,np.nan]
                    ds['artFFT'].append(art_result[0])
                    ds['artNK1'].append(art_result[1])
                else:
                    ds['artFFT'].append(np.nan)
                    ds['artNK1'].append(np.nan)
            
            else:
                ds['artFFT'].append(np.nan)
                ds['artNK1'].append(np.nan)
                
            #Pleth Features:plethProcessing(wf,fs
            if exists['pleth']:
                data['pleth'] = np.array(h5py['pleth'][(h5py['pleth']['time'] > time-basetime) & (h5py['pleth']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['pleth'],FS['pleth']):
                    try:
                        pleth_result = plethProcessingRR(data['pleth'],FS['pleth'])
                    except:
                        pleth_result = [[np.nan,np.nan],np.nan]
                    ds['plethNK1'].append(pleth_result[0])
                    ds['plethFFT'].append(pleth_result[1])
                else:
                    ds['plethNK1'].append([np.nan,np.nan])
                    ds['plethFFT'].append(np.nan)
            
            else:
                ds['plethNK1'].append([np.nan,np.nan])
                ds['plethFFT'].append(np.nan)
                
            
            #PlethT Features:plethProcessing(wf,fs
            if exists['plethT']:
                data['plethT'] = np.array(h5py['plethT'][(h5py['plethT']['time'] > time-basetime) & (h5py['plethT']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['plethT'],FS['plethT']):
                    try:
                        plethT_result = plethProcessingRR(data['plethT'],FS['plethT'])
                    except:
                        plethT_result = [[np.nan,np.nan],np.nan]
                    ds['plethTNK1'].append(plethT_result[0])
                    ds['plethTFFT'].append(plethT_result[1])
                else:
                    ds['plethTNK1'].append([np.nan,np.nan])
                    ds['plethTFFT'].append(np.nan)
            
            else:
                ds['plethTNK1'].append([np.nan,np.nan])
                ds['plethTFFT'].append(np.nan)
                
                
            #ECGii Features:ecgProcessing(wf,FS)
            if exists['ecgii']:
                data['ecgii'] = np.array(h5py['ecgii'][(h5py['ecgii']['time'] > time-basetime) & (h5py['ecgii']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['ecgii'],FS['ecgii']):
                    try:
                        ecgii_result = ecgProcessing(data['ecgii'],FS['ecgii'])
                    except:
                        ecgii_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['ecgiiFFT'].append(ecgii_result[0])
                    ds['ecgiiINT'].append(ecgii_result[1])
                    ds['ecgiiNK1'].append(ecgii_result[2])
                    ds['ecgiiNK2'].append(ecgii_result[3])
                    
                else:
                    ds['ecgiiFFT'].append(np.nan)
                    ds['ecgiiINT'].append(np.nan)
                    ds['ecgiiNK1'].append(np.nan)
                    ds['ecgiiNK2'].append(np.nan)
            
            else:
                ds['ecgiiFFT'].append(np.nan)
                ds['ecgiiINT'].append(np.nan)
                ds['ecgiiNK1'].append(np.nan)
                ds['ecgiiNK2'].append(np.nan)
            
            #ECGiii Features:ecgProcessing(wf,FS)
            if exists['ecgiii']:
                data['ecgiii'] = np.array(h5py['ecgiii'][(h5py['ecgiii']['time'] > time-basetime) & (h5py['ecgiii']['time'] < time+60-basetime)]['value'])
                if wf_exists(data['ecgiii'],FS['ecgiii']):
                    try:
                        ecgiii_result = ecgProcessing(data['ecgiii'],FS['ecgiii'])
                    except:
                        ecgiii_result = [np.nan,np.nan,np.nan,np.nan]
                    ds['ecgiiiFFT'].append(ecgiii_result[0])
                    ds['ecgiiiINT'].append(ecgiii_result[1])
                    ds['ecgiiiNK1'].append(ecgiii_result[2])
                    ds['ecgiiiNK2'].append(ecgiii_result[3])
                    
                else:
                    ds['ecgiiiFFT'].append(np.nan)
                    ds['ecgiiiINT'].append(np.nan)
                    ds['ecgiiiNK1'].append(np.nan)
                    ds['ecgiiiNK2'].append(np.nan)
            
            else:
                ds['ecgiiiFFT'].append(np.nan)
                ds['ecgiiiINT'].append(np.nan)
                ds['ecgiiiNK1'].append(np.nan)
                ds['ecgiiiNK2'].append(np.nan)
       
        #Final Assignment of all variables into respective DF column    
        subResult = pd.DataFrame([ds])        
        individualAlerts.append(subResult)      
    result = pd.concat(individualAlerts)
    
    #change index result df with indices of subAlert
    result.set_index(match_ind,inplace=True)
    result.to_pickle(pathToSaveData + f"/{fName[:-3]}.pkl")        