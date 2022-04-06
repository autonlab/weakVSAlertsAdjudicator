import pandas as pd
import numpy as np
from statistics import mean
import scipy as ss
from scipy.stats import t

from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from sklearn.metrics import roc_curve, roc_auc_score, auc
from bokeh.palettes import Category10
output_notebook()



def ROC_Curve_Figure_Setup():
    plot_linear = figure(x_axis_label='FPR', y_axis_label='TPR', x_range=(0, 1), y_range=(0, 1.01))
    plot_linear_10_percent = figure(x_axis_label='FPR', y_axis_label='TPR', x_range=(0, 0.10), y_range=(0, 1.01))
    plot_log_fpr = figure(x_axis_label='log10(FPR)', y_axis_label='TPR', x_range=(-4, 0), y_range=(0, 1.01))
    plot_log_fnr = figure(x_axis_label='log10(FNR)', y_axis_label='TNR', x_range=(-4, 0), y_range=(0, 1.01))

    rand_seq = np.linspace(0, 1, 10000)[1:-1]
    plot_linear.line(rand_seq, rand_seq, color='black', legend_label='Random', line_width = 3)
    plot_log_fpr.line(np.log10(rand_seq), rand_seq, color='black', line_width = 3)
    plot_log_fnr.line(np.log10(rand_seq), rand_seq, color='black', line_width = 3)
    
    plot_linear.xaxis.axis_label_text_font_size = '20pt'
    plot_linear.xaxis.major_label_text_font_size = '14pt'
    plot_linear.yaxis.axis_label_text_font_size = '20pt'
    plot_linear.yaxis.major_label_text_font_size = '14pt'
    
    plot_linear_10_percent.xaxis.axis_label_text_font_size = '20pt'
    plot_linear_10_percent.xaxis.major_label_text_font_size = '14pt'
    plot_linear_10_percent.yaxis.axis_label_text_font_size = '20pt'
    plot_linear_10_percent.yaxis.major_label_text_font_size = '14pt'
    
    plot_log_fpr.xaxis.axis_label_text_font_size = '20pt'
    plot_log_fpr.xaxis.major_label_text_font_size = '14pt'
    plot_log_fpr.yaxis.axis_label_text_font_size = '20pt'
    plot_log_fpr.yaxis.major_label_text_font_size = '14pt'
    
    plot_log_fnr.xaxis.axis_label_text_font_size = '20pt'
    plot_log_fnr.xaxis.major_label_text_font_size = '14pt'
    plot_log_fnr.yaxis.axis_label_text_font_size = '20pt'
    plot_log_fnr.yaxis.major_label_text_font_size = '14pt'
    return(plot_linear, plot_linear_10_percent, plot_log_fpr, plot_log_fnr)

    
    
    

def CI_Bounds(N_Folds, y_mu, y_se, fraction, support):
    if N_Folds != 1:
        Critical_Value = abs(t.ppf(0.025, N_Folds-1))
        se_scalar = np.sqrt(N_Folds)
        LB = y_mu-Critical_Value*(y_se/se_scalar)
        UB = y_mu+Critical_Value*(y_se/se_scalar)
    else:
        LB = (1/(1+1.96**2/support))*(fraction + 1.96**2/(2*support)) - (1.96/(1+1.96**2/support))*(np.sqrt(fraction*(1-fraction)/support + 1.96**2/(4*support**2))) # Wilsons Score CI
        UB = (1/(1+1.96**2/support))*(fraction + 1.96**2/(2*support)) + (1.96/(1+1.96**2/support))*(np.sqrt(fraction*(1-fraction)/support + 1.96**2/(4*support**2)))
    return(LB, UB)





def Plot_Curve(scores_for_plot, Color, str_model_name, plot_linear, plot_linear_10_percent, plot_log_fpr, plot_log_fnr):
    Number_Folds = len(scores_for_plot)
    if Number_Folds == 0: print('error: Number of folds must be >=1')
    COLOR = Color 
    model_name = str_model_name
    log_min=-4
    
    df_roc_all = []
    ars_all = []
    for i in range(Number_Folds):
        ID = scores_for_plot[i].copy()
        Ytest = ID['truth']
        htest = ID['positive']
        fpr, tpr, thres = roc_curve(Ytest, htest)
        ars = roc_auc_score(Ytest, htest)
        ars_all.append(ars)
        df_roc = pd.DataFrame({
                    'fpr': fpr,
                    'fnr': 1 - tpr,
                    'tpr': tpr,
                    'tnr': 1 - fpr,
                    'threshold': thres})
        df_roc_all.append(df_roc)



    agg_df_linear = []
    agg_df_logfpr = []
    agg_df_logfnr = []

    for fi, df in enumerate(df_roc_all):
        x = df_roc_all[fi]['fpr'].values
        y = df_roc_all[fi]['tpr'].values
        x_label = 'fpr'
        y_label = 'tpr'
        x_values = np.linspace(0, 1, 500)

        y_interp = np.interp(x_values, x, y, left=0, right=1)
        agg_df_linear.append(pd.DataFrame({x_label: x_values, y_label: y_interp}))
    agg_df_linear = pd.concat(agg_df_linear).groupby(x_label).agg([np.mean, np.std])
    x_linear = agg_df_linear.index.values
    y_mu_linear = agg_df_linear['tpr']['mean'].values
    y_sd_linear = agg_df_linear['tpr']['std'].values
    Linear_LB, Linear_UB = CI_Bounds(Number_Folds, y_mu_linear, y_sd_linear, y_interp, len(ID))
    tpr_1 = agg_df_linear.loc[0.01002004008016032]['tpr']['mean'].round(4)   
    auc_score = auc(agg_df_linear.index.values, agg_df_linear['tpr']['mean'].values)
    
    plot_linear.line(x_linear, y_mu_linear, color = COLOR, line_width = 4, legend_label = 'Mean ROC_AUC = %0.3f - %s' %(mean(ars_all), model_name))
    plot_linear.varea(x_linear, Linear_LB, Linear_UB, alpha=0.1, color = COLOR)
    
    plot_linear_10_percent.line(x_linear, y_mu_linear, color = COLOR, line_width = 4, legend_label = 'TPR at 0.01 FPR = %0.3f - %s' %(tpr_1, model_name))
    plot_linear_10_percent.varea(x_linear, Linear_LB, Linear_UB, alpha=0.1, color = COLOR)



    for fi, df in enumerate(df_roc_all):
        x = df_roc_all[fi]['fpr'].values
        y = df_roc_all[fi]['tpr'].values
        x_label = 'fpr'
        y_label = 'tpr'
        x_values = 10**np.linspace(log_min, 0, 500)

        y_interp = np.interp(x_values, x, y, left=0, right=1)
        agg_df_logfpr.append(pd.DataFrame({x_label: x_values, y_label: y_interp}))
    agg_df_logfpr = pd.concat(agg_df_logfpr).groupby(x_label).agg([np.mean, np.std])
    
    x_logfpr = np.log10(agg_df_logfpr.index.values)
    y_mu_logfpr = agg_df_logfpr['tpr']['mean'].values
    y_sd_logfpr = agg_df_logfpr['tpr']['std'].values
    logfpr_LB, logfpr_UB = CI_Bounds(Number_Folds, y_mu_logfpr, y_sd_logfpr, y_interp, len(ID))

    plot_log_fpr.line(x_logfpr, y_mu_logfpr, color = COLOR, line_width = 4, legend_label = 'TPR at 0.01 FPR = %0.3f - %s' %(tpr_1, model_name))
    plot_log_fpr.varea(x_logfpr, logfpr_LB, logfpr_UB, alpha=0.1, color = COLOR)


    for fi, df in enumerate(df_roc_all):
        x = df_roc_all[fi]['fnr'].values
        x = x[::-1]
        y = df_roc_all[fi]['tnr'].values
        y = y[::-1]
        x_label = 'fnr'
        y_label = 'tnr'
        x_values = 10**np.linspace(log_min, 0, 500)

        y_interp = np.interp(x_values, x, y, left=0, right=1)
        agg_df_logfnr.append(pd.DataFrame({x_label: x_values, y_label: y_interp}))
    agg_df_logfnr = pd.concat(agg_df_logfnr).groupby(x_label).agg([np.mean, np.std])
    x_logfnr = np.log10(agg_df_logfnr.index.values)
    y_mu_logfnr = agg_df_logfnr['tnr']['mean'].values
    y_sd_logfnr = agg_df_logfnr['tnr']['std'].values
    logfnr_LB, logfnr_UB = CI_Bounds(Number_Folds, y_mu_logfnr, y_sd_logfnr, y_interp, len(ID))
    tnr_1 = agg_df_logfnr.loc[0.010092715146305707]['tnr']['mean'].round(4)

    plot_log_fnr.line(x_logfnr, y_mu_logfnr, color = COLOR, line_width = 4, legend_label = 'TNR at 0.01 FNR =  %0.3f - %s' %(tnr_1, model_name))
    plot_log_fnr.varea(x_logfnr, logfnr_LB, logfnr_UB, alpha=0.1, color = COLOR)


    
    
    
def Show_ROC_Curves(plot_linear, plot_linear_10_percent, plot_log_fpr, plot_log_fnr, show_legend = True):
    if (show_legend == False):
        plot_linear.legend.visible = False
        plot_linear_10_percent.legend.visible = False
        plot_log_fpr.legend.visible = False
        plot_log_fnr.legend.visible = False
        show(plot_linear)
        show(plot_linear_10_percent)
        show(plot_log_fpr)
        show(plot_log_fnr)
    else:
        plot_linear.legend.location = 'bottom_right'
        plot_linear.legend.label_text_font_size = "12pt"

        plot_linear_10_percent.legend.location = 'bottom_right'
        plot_linear_10_percent.legend.label_text_font_size = "12pt"

        plot_log_fpr.legend.location = 'bottom_right'
        plot_log_fpr.legend.label_text_font_size = "12pt"

        plot_log_fnr.legend.location = 'bottom_right'
        plot_log_fnr.legend.label_text_font_size = "12pt"

        show(plot_linear)
        show(plot_linear_10_percent)
        show(plot_log_fpr)
        show(plot_log_fnr)