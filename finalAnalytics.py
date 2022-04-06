import Plot_ROC_Curves as prcc
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from sklearn.metrics import roc_curve, roc_auc_score, auc
from bokeh.palettes import Category10

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from snorkel.labeling import filter_unlabeled_dataframe
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.inspection import permutation_importance

from snorkel.labeling.model import LabelModel
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds

from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Models
rr_WS = pd.read_pickle('../RISS Project/RF_preds_sorted.pkl')
rr_ML = pd.read_pickle('../RISS Project/RF_preds_sorted_majlab.pkl')
rr_SL = pd.read_pickle('../RISS Project/RF_preds_sorted_supervised.pkl')
rr_PL = pd.read_pickle('../RISS Project/PL_preds_sorted.pkl')
models_rr = [rr_WS, rr_ML, rr_SL,rr_PL]

rr_WS_abl = pd.read_pickle('../RISS Project/RF_preds_sorted_abellation.pkl')
rr_SL_abl = pd.read_pickle('../RISS Project/RF_preds_sorted_supervised_abellation.pkl')
models_rr_abl = [rr_WS, rr_WS_abl, rr_SL, rr_SL_abl, rr_PL]



sp_WS = pd.read_pickle('../RISS Project/RF_preds_sorted_spo2.pkl')
sp_ML = pd.read_pickle('../RISS Project/RF_preds_sorted_majlab_spo2.pkl')
sp_SL = pd.read_pickle('../RISS Project/RF_preds_sorted_supervised_spo2.pkl')
sp_PL = pd.read_pickle('../RISS Project/PL_preds_sorted_spo2.pkl')
models_sp = [sp_WS, sp_ML, sp_SL, sp_PL]

sp_WS_abl = pd.read_pickle('../RISS Project/RF_preds_sorted_spo2_abellation.pkl') #Yes, it is supposed to be spelled ablation
# sp_ML_abl = pd.read_pickle('../RISS Project/RF_preds_sorted_majlab_spo2_abellation.pkl')
sp_SL_abl = pd.read_pickle('../RISS Project/RF_preds_sorted_supervised_spo2_abellation.pkl')
models_sp_abl = [sp_WS, sp_WS_abl, sp_SL, sp_SL_abl, sp_PL]


def aggregateHelper(modelPredictions, metrics = False):
    list = []
    df = pd.DataFrame()
    df['positive'] = modelPredictions.positive_proba
    df['truth'] = modelPredictions.ground_truth
    list.append(df)

    if (metrics):
        fpr,tpr,ths = roc_curve(df['truth'],df['positive'],drop_intermediate=False)
        pumetrics = pd.DataFrame(np.transpose([fpr,tpr,ths]), columns = ['fpr','tpr','ths'])
        pumetrics['fnr'] = 1 - pumetrics.tpr.values
        pumetrics['tnr'] = 1 - pumetrics.fpr.values
        metr = pd.DataFrame()
        metr = pd.DataFrame()
        metr['tpr_at_fprpoint01'] = pd.Series(pumetrics.tpr.values[np.argmin(np.abs(pumetrics.fpr.values - 0.01))])
        metr['tnr_at_fnrpoint01'] = pd.Series(pumetrics.tnr.values[np.argmin(np.abs(pumetrics.fnr.values - 0.01))])
        metr['fpr_at_tprpoint5'] = pd.Series(pumetrics.fpr.values[np.argmin(np.abs(pumetrics.tpr.values - 0.5))])
        metr['fnr_at_tnrpoint5'] = pd.Series(pumetrics.fnr.values[np.argmin(np.abs(pumetrics.tnr.values - 0.5))])
        metr['auc'] = pd.Series(roc_auc_score(df['truth'], df['positive']))
    return list if not metrics else metr

def aggregate(modelArray):
    masterlist = []
    for model in modelArray:
        masterlist.append(aggregateHelper(model))
    return masterlist


def plotROC(modelArray, Model_Names = ['RF - WS','RF - Majority Labeler','RF - Supervised', 'Probability Labels'], colors = ['red','green','blue','gray'], showLegend = False):
    masterlist = aggregate(modelArray)
    p_linear, p_linear_10_percent, p_pos_log, p_neg_log = prcc.ROC_Curve_Figure_Setup()
    for i in range(len(masterlist)):
        prcc.Plot_Curve(masterlist[i], colors[i], Model_Names[i], p_linear, p_linear_10_percent, p_pos_log, p_neg_log)
    prcc.Show_ROC_Curves(p_linear, p_linear_10_percent, p_pos_log, p_neg_log, show_legend = showLegend)

def performanceMetrics(predictions, model_name):
    print(model_name + ': \n')
    print(classification_report(predictions.ground_truth, predictions.predicted,digits=4))
    print(aggregateHelper(predictions, True))
    
    cm = confusion_matrix(predictions.ground_truth, predictions.predicted, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
    disp.plot()
    plt.show()

