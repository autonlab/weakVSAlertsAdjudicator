import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier
from snorkel.utils import probs_to_preds
from tqdm import tqdm
from snorkel.labeling.model import LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling.model import MajorityLabelVoter
import utils as ut

def getOrder(ind, pid):
    order = pd.DataFrame()
    order['ind'] = ind
    order['patientID'] = pid
    return order


def trainTest_WS(X, y, order, lfs, X_drop = None):
    if X_drop is None:
        X_drop = X


    logo = LeaveOneGroupOut()
    #Custom Cross Validation

    ##List of DataFrames
    truth = []
    importance = []
    acc = []
    for train, test in tqdm(logo.split(X, y, groups=order['patientID'])):
        #Data Organization
        truthKeys = ['ind','patientID','positive_proba','predicted','ground_truth']
        di1 = dict.fromkeys(truthKeys)
        fiKeys = ['feature_importance_attribute','permutation_feature_importance']
        di2 = dict.fromkeys(fiKeys)
        
        #index
        di1['ind'] = test
        di1['patientID'] = order.loc[test,'patientID']
        
        
        
        #Snorkel LabelModel
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df = X.iloc[train,:])
        
        label_model = LabelModel(cardinality=2, verbose=True) #class_balance = np.array([0.25,.75]) # Y_dev = balance
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100,class_balance = np.array([0.25,.75]), seed=123,lr = .005)
        
        probs_train = label_model.predict_proba(L_train)
        
        #Filtered Snorkel Labels
        X_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=X_drop.iloc[train,:], y=probs_train, L=L_train)
        y_train_filtered = probs_to_preds(probs=probs_train_filtered)
        
        #Random Forest Model
        num_trees = 1000
        RFModel = RandomForestClassifier(n_estimators = num_trees,max_depth = 5,class_weight='balanced',n_jobs=5,random_state = 7 )
        RFModel.fit(X=X_train_filtered,y=y_train_filtered)
        di1['predicted'] = RFModel.predict(X_drop.iloc[test,:])
        di1['positive_proba'] = ut.positive_proba_binary(RFModel, X_drop.iloc[test,:])
        di1['ground_truth'] = y[test]
        
        # Feature Importance
    #     di2['feature'] = X_train_filtered.columns.values
        di2['feature_importance_attribute'] = RFModel.feature_importances_
        
        permImportance = permutation_importance(RFModel,X_train_filtered,y_train_filtered, n_repeats = 10,scoring = 'f1')
        di2['permutation_feature_importance'] = permImportance['importances_mean']
        
        #Dict2DF Conversion
        truthEl = pd.DataFrame(di1)
        fiEl = pd.DataFrame(di2)
        
        #append to list
        truth.append(truthEl)
        importance.append(fiEl)

    #Post-processing Stuff
    preds = pd.concat(truth)
    preds_sort = preds.sort_values(by=['ind'])

    imp = pd.concat(importance)
    fi = imp.groupby(imp.index)
    fi_mean = fi.mean()
    # fi_median = fi.median()

    fia = pd.DataFrame()
    fia['feature'] = X_drop.columns.values
    fia['feature_importance_attribute'] = fi_mean.feature_importance_attribute.values
    fia_sorted = fia.sort_values(by=['feature_importance_attribute'],ascending=False)
    fias = fia_sorted.reset_index(drop=True)

    pfi = pd.DataFrame()
    pfi['feature'] = X_drop.columns.values
    pfi['permutation_feature_importance'] = fi_mean.permutation_feature_importance.values
    pfi_sorted = pfi.sort_values(by=['permutation_feature_importance'],ascending=False)
    pfis = pfi_sorted.reset_index(drop=True)

    return [preds_sort, fias, pfis]

def trainTest_FS(X, y, order, lfs, X_drop = None):
    if X_drop is None:
        X_drop = X


    logo = LeaveOneGroupOut()

    #Custom Cross Validation

    ##List of DataFrames
    truth = []
    importance = []
    acc = []
    for train, test in tqdm(logo.split(X_drop, y, groups=order['patientID'])):
        #Data Organization
        truthKeys = ['ind','patientID','positive_proba','predicted','ground_truth']
        di1 = dict.fromkeys(truthKeys)
        fiKeys = ['feature_importance_attribute','permutation_feature_importance']
        di2 = dict.fromkeys(fiKeys)
        
        #index
        di1['ind'] = test
        di1['patientID'] = order.loc[test,'patientID']
        
        X_train_filtered = X_drop.iloc[train,:]
        y_train_filtered = y[train]
        
        #Random Forest Model
        num_trees = 1000
        RFModel = RandomForestClassifier(n_estimators = num_trees,max_depth = 5,class_weight='balanced',n_jobs=5,random_state = 7 )
        RFModel.fit(X=X_train_filtered,y=y_train_filtered)
        di1['predicted'] = RFModel.predict(X_drop.iloc[test,:])
        di1['positive_proba'] = ut.positive_proba_binary(RFModel, X_drop.iloc[test,:])
        di1['ground_truth'] = y[test]
        
        # Feature Importance
        di2['feature_importance_attribute'] = RFModel.feature_importances_
        
        permImportance = permutation_importance(RFModel,X_train_filtered,y_train_filtered, n_repeats = 10,scoring = 'f1')
        di2['permutation_feature_importance'] = permImportance['importances_mean']
        
        #Dict2DF Conversion
        truthEl = pd.DataFrame(di1)
        fiEl = pd.DataFrame(di2)
        
        #append to list
        truth.append(truthEl)
        importance.append(fiEl)

    #Post-processing Stuff
    preds = pd.concat(truth)
    preds_sort = preds.sort_values(by=['ind'])

    imp = pd.concat(importance)
    fi = imp.groupby(imp.index)
    fi_mean = fi.mean()
    # fi_median = fi.median()

    fia = pd.DataFrame()
    fia['feature'] = X_drop.columns.values
    fia['feature_importance_attribute'] = fi_mean.feature_importance_attribute.values
    fia_sorted = fia.sort_values(by=['feature_importance_attribute'],ascending=False)
    fias = fia_sorted.reset_index(drop=True)

    pfi = pd.DataFrame()
    pfi['feature'] = X_drop.columns.values
    pfi['permutation_feature_importance'] = fi_mean.permutation_feature_importance.values
    pfi_sorted = pfi.sort_values(by=['permutation_feature_importance'],ascending=False)
    pfis = pfi_sorted.reset_index(drop=True)

    return [preds_sort, fias, pfis]

def trainTest_MajLab(X, y, order, lfs, X_drop = None):
    if X_drop is None:
        X_drop = X

    logo = LeaveOneGroupOut()

    #Custom Cross Validation

    ##List of DataFrames
    truth = []
    importance = []
    acc = []
    for train, test in tqdm(logo.split(X, y, groups=order['patientID'])):
        #Data Organization
        truthKeys = ['ind','patientID','positive_proba','predicted','ground_truth']
        di1 = dict.fromkeys(truthKeys)
        fiKeys = ['feature_importance_attribute','permutation_feature_importance']
        di2 = dict.fromkeys(fiKeys)
        
        #index
        di1['ind'] = test
        di1['patientID'] = order.loc[test,'patientID']
        
        ##########Snorkel Baseline Majority Model########
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df = X.iloc[train,:])
        majority_model = MajorityLabelVoter()
        probs_train = majority_model.predict_proba(L_train)
        
        #Filtered Snorkel Labels
        X_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=X_drop.iloc[train,:], y=probs_train, L=L_train)
        y_train_filtered = probs_to_preds(probs=probs_train_filtered)
        
        #Random Forest Model
        num_trees = 1000
        RFModel = RandomForestClassifier(n_estimators = num_trees,max_depth = 5,class_weight='balanced',n_jobs=5,random_state = 7 )
        RFModel.fit(X=X_train_filtered,y=y_train_filtered)
        di1['predicted'] = RFModel.predict(X_drop.iloc[test,:])
        di1['positive_proba'] = ut.positive_proba_binary(RFModel, X_drop.iloc[test,:])
        di1['ground_truth'] = y[test]
        
        # Feature Importance
    #     di2['feature'] = X_train_filtered.columns.values
        di2['feature_importance_attribute'] = RFModel.feature_importances_
        
        permImportance = permutation_importance(RFModel,X_train_filtered,y_train_filtered, n_repeats = 10, scoring = 'f1')
        di2['permutation_feature_importance'] = permImportance['importances_mean']
        
        #Dict2DF Conversion
        truthEl = pd.DataFrame(di1)
        fiEl = pd.DataFrame(di2)
        
        #append to list
        truth.append(truthEl)
        importance.append(fiEl)

    #Post-processing Stuff
    preds = pd.concat(truth)
    preds_sort = preds.sort_values(by=['ind'])

    imp = pd.concat(importance)
    fi = imp.groupby(imp.index)
    fi_mean = fi.mean()
    # fi_median = fi.median()

    fia = pd.DataFrame()
    fia['feature'] = X_drop.columns.values
    fia['feature_importance_attribute'] = fi_mean.feature_importance_attribute.values
    fia_sorted = fia.sort_values(by=['feature_importance_attribute'],ascending=False)
    fias = fia_sorted.reset_index(drop=True)

    pfi = pd.DataFrame()
    pfi['feature'] = X_drop.columns.values
    pfi['permutation_feature_importance'] = fi_mean.permutation_feature_importance.values
    pfi_sorted = pfi.sort_values(by=['permutation_feature_importance'],ascending=False)
    pfis = pfi_sorted.reset_index(drop=True)

    return [preds_sort, fias, pfis]

def trainTest_ProbLab(X, y, order, lfs): #PROBABILITY LABELS don't have ablation settings
    logo = LeaveOneGroupOut()

    #Custom Cross Validation

    ##List of DataFrames
    truth = []
    acc = []
    for train, test in logo.split(X, y, groups=order['patientID']):
        #Data Organization
        truthKeys = ['ind','patientID','positive_proba','predicted','ground_truth']
        di1 = dict.fromkeys(truthKeys)

        
        #index
        di1['ind'] = test
        di1['patientID'] = order.loc[test,'patientID']
        
        
        
        #Snorkel LabelModel
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df = X.iloc[train,:])
        L_test = applier.apply(df=X.iloc[test,:])
        
        label_model = LabelModel(cardinality=2, verbose=True) #class_balance = np.array([0.25,.75]) # Y_dev = balance
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100,class_balance = np.array([0.25,.75]), seed=123,lr = .005)
        
        probs_dev = label_model.predict_proba(L_test)
        preds_dev = probs_to_preds(probs_dev)
        

        di1['predicted'] = preds_dev
        di1['positive_proba'] = probs_dev[:,1]
        di1['ground_truth'] = y[test]
        

        #Dict2DF Conversion
        truthEl = pd.DataFrame(di1)
        
        #append to list
        truth.append(truthEl)

    #Post processing stuff
    preds = pd.concat(truth)
    preds_sort = preds.sort_values(by=['ind'])

    return preds_sort