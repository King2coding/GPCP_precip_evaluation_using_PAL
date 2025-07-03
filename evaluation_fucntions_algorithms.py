# functions to evaluate models using both categorical and quantitative statistics
# functions largely rely on the https://hydroerr.readthedocs.io/_/downloads/en/stable/pdf/ package 

# import packages
import pandas as pd
import numpy as np
import HydroErr as he



# the categorical metrics
#%%
def binary_cat_metrics(df, test_label, pred_label,target_val,class_cat):
    '''
    Binary_cat_metrics fucntions is for computing categroical mertrics to evaluate classification models
    Can be used for evaluating multiclassification cases but this has to be applied for each class

    requires:
    df = dataframe containing observed and predicted labels

    test_label = columns name of the observed (test) data

    pred_label = columns name of the predictted/simulated (model preicted) data

    target value  = the value/class label in the observed to be evaluated 

    class_cat = the category/class been evaluated

    returns a dataframe of catagorical metrics Hits, Miss, False alarms, POD, FAR, POFD,
    ACC, CSI, HSS, ETS, HK

    since a multicalss is evaliuated this has to be run for each class
    '''

    df_cat = pd.DataFrame()
    
    a_hit = df.loc[(df[test_label] == target_val)  & (df[pred_label] == target_val)].count()[0]

    b_false = df.loc[(df[test_label] != target_val) & (df[pred_label] == target_val)].count()[0]

    c_miss = df.loc[(df[test_label] == target_val) & (df[pred_label] != target_val)].count()[0]

    # d_cor_neg = df.loc[(df[test_label] != target_val) & (df[pred_label] != target_val)].count()[0]
    # d_cor_neg = df.loc[((df[test_label] < target_val) & (df[pred_label] < target_val))].count()[0]

    # n = a_hit + b_false + c_miss + d_cor_neg # total number of calssifications

    #a_hit_rand_hss = ((a_hit + b_miss)*(a_hit + c_false) + (d_cor_false + b_miss)*(d_cor_false + c_false))/N # random hits for HSS

    #a_hit_rand_ets = (a_hit + b_miss)/N # random hits for ETS
    # a_ref = (a_hit + b_false)*(a_hit + c_miss) / n

    # TOB = a_hit + b_miss # total number of true classifications

    df_cat.loc[class_cat ,'Hits'] = a_hit #round((a_hit/TOB)*100,1) 

    df_cat.loc[class_cat ,'Miss'] = c_miss #round((c_miss/TOB)*100,1)

    df_cat.loc[class_cat,'False alarms'] = b_false #round((c_false/TOB)*100,1)

    df_cat.loc[class_cat ,'POD'] = round(a_hit/(a_hit + c_miss),3)

    df_cat.loc[class_cat,'FAR'] = round(b_false/(a_hit + b_false),3)

    # df_cat.loc[class_cat,'POFD'] = round(b_false/(d_cor_neg + b_false),3)

    # df_cat.loc[class_cat,'ACC'] = round((a_hit + d_cor_neg)/ n,2)

    df_cat.loc[class_cat,'Bias'] = round((a_hit + b_false)/(a_hit + c_miss),2)

    df_cat.loc[class_cat,'CSI'] = round(a_hit/(a_hit + b_false + c_miss),2)

    # df_cat.loc[class_cat,'HSS'] = round((2*((a_hit*d_cor_neg) - (b_false*c_miss)))/(((a_hit+c_miss)*(c_miss+d_cor_neg))+((a_hit+b_false)*(b_false+d_cor_neg))),2)
    #round(((a_hit + d_cor_false)-(a_hit_rand_hss))/(N - a_hit_rand_hss),3)

    # df_cat.loc[class_cat,'ETS'] = round((a_hit - a_ref)/(a_hit - a_ref + b_false + c_miss),2) 
    #round((a_hit - a_hit_rand_ets)/(a_hit + b_miss + c_false - a_hit_rand_ets),3)   

    return df_cat
#***********************************************************************************************************************************************************
# df based
def continous_cat_metrics(df, test_label, pred_label,target_val,class_name):
    '''
    continous_cat_metrics is for computoing categorical metrics for evaluating continous data

    df = dataframe containing observed and predicted labels

    test_label = columns name of the observed (test) data

    pred_label = columns name of the predictted (model preicted) data

    target value  = the value/class label in the observed to be evaluated 

    class_name = an arbitrary name to be used as index of the dataframe this function returns

    returns a dataframe of catagorical metrics Hits, Miss, False alarms, POD, FAR, POFD,
    ACC, CSI, HSS, ETS, HK

    since a multicalss is evaliuated this has to be run for each class
    '''

    df_cat = pd.DataFrame()
    
    a_hit = df.loc[(df[test_label] >= target_val)  & (df[pred_label] >= target_val)].count()[0]

    b_miss = df.loc[(df[test_label] >= target_val) & (df[pred_label] < target_val)].count()[0]

    c_false = df.loc[(df[test_label] < target_val) & (df[pred_label] >= target_val)].count()[0]

    # d_cor_false = df.loc[(df[test_label] < target_val) & (df[pred_label] < target_val)].count()[0]
    d_cor_false = df.loc[((df[test_label] < target_val) & (df[pred_label] < target_val))].count()[0]

    N = a_hit + b_miss + c_false + d_cor_false # total number of calssifications

    a_hit_rand_hss = ((a_hit + b_miss)*(a_hit + c_false) + (d_cor_false + b_miss)*(d_cor_false + c_false))/N # random hits for HSS

    a_hit_rand_ets = (a_hit + b_miss)/N # random hits for ETS

    TOB = a_hit + b_miss # total true estimates

    df_cat.loc[class_name ,'Hits'] = round((a_hit/TOB)*100,1) 

    df_cat.loc[class_name ,'Miss'] = round((b_miss/TOB)*100,1)

    df_cat.loc[class_name,'False alarms'] = round((c_false/TOB)*100,1)

    df_cat.loc[class_name ,'POD'] = round(a_hit/(a_hit + b_miss),3)

    df_cat.loc[class_name,'FAR'] = round(c_false/(a_hit + c_false),3)

    df_cat.loc[class_name,'POFD'] = round(c_false/(d_cor_false + c_false),3)

    df_cat.loc[class_name,'ACC'] = round((a_hit + d_cor_false)/ N,3)

    df_cat.loc[class_name,'CSI'] = round(a_hit/(a_hit + b_miss + c_false),3)

    df_cat.loc[class_name,'HSS'] = round(((a_hit + d_cor_false)-(a_hit_rand_hss))/(N - a_hit_rand_hss),3)

    df_cat.loc[class_name,'ETS'] = round((a_hit - a_hit_rand_ets)/(a_hit + b_miss + c_false - a_hit_rand_ets),3)

    # Hanssen and Kuipers discriminant (true skill statistic, Peirce's skill score)
    df_cat.loc[class_name,'HK'] = round((a_hit/a_hit + b_miss) - (c_false/d_cor_false + c_false),3)

    return df_cat

#***********************************************************************************************************************************************************

# array based
def continous_cat_metrics_array_based(obs, sim,target_val):
    '''
    continous_cat_metrics_array_based is is an array based implementation of continous_cat_metrics, which is df based

    it requires 
    obs = observation (x variable)
    sim = simulated (y variable)   

    target value  = the value/class label in the observed to be evaluated 

    returns a dataframe of catagorical metrics Hits, Miss, False alarms, POD, FAR, POFD,
    ACC, CSI, HSS, ETS

    These computations were aquird from Wilks, D.S. Statistical Methods in the Atmospheric Sciences; Elsevier: Amsterdam, The Netherlands, 2011; p. 627.
    '''

    if target_val == 0:
        a_hit = np.count_nonzero((obs > target_val) & (sim > target_val))

        b_false = np.count_nonzero((obs == target_val) & (sim > target_val))

        c_miss = np.count_nonzero((obs > target_val) & (sim == target_val))  

        d_cor_neg = np.count_nonzero((obs == target_val) & (sim == target_val))

    elif target_val > 0:

        a_hit = np.count_nonzero((obs > target_val) & (sim > target_val))

        b_false = np.count_nonzero((obs < target_val) & (sim > target_val))

        c_miss = np.count_nonzero((obs > target_val) & (sim < target_val))  

        d_cor_neg = np.count_nonzero((obs < target_val) & (sim < target_val))

    n = a_hit + b_false + c_miss + d_cor_neg # total number of calssifications

    # a_hit_rand_hss = ((a_hit + c_miss)*(a_hit + b_false) + (d_cor_neg + c_miss)*(d_cor_neg + b_false))/n # random hits for HSS

    #a_hit_rand_ets = (a_hit + c_miss)*(a_hit + b_false)/n # random hits for ETSS   
    # a_ref = (a_hit + b_false)*(a_hit + c_miss) / n    

    # Calculate metrics with safe handling for division by zero
    pod = round(a_hit / (a_hit + c_miss), 2) if (a_hit + c_miss) > 0 else np.nan
    far = round(b_false / (a_hit + b_false), 2) if (a_hit + b_false) > 0 else np.nan
    bias = round((a_hit + b_false) / (a_hit + c_miss), 2) if (a_hit + c_miss) > 0 else np.nan
    csi = round(a_hit / (a_hit + b_false + c_miss), 2) if (a_hit + b_false + c_miss) > 0 else np.nan


    # pod = round(a_hit/(a_hit + c_miss),2)
    # the hit rate is the ratio of correct forecasts to the number of times this event occurred
    # range  between 0 (worse) to 1 (perfect)

    # far = round(b_false/(a_hit + b_false),2)
    # That is, FAR is the fraction of yes forecasts that turn out to be wrong, or that proportion of the forecast events that fail to materialize
    # range  between 0 (perfect) to 1 (worse case)

    # pofd = round(b_false/(b_false + d_cor_neg),2)
    # ratio of false alarms to the total number of nonoccurrences of the event

    # bias = round((a_hit + b_false)/(a_hit + c_miss),2)
    # bias is simply the ratio of the number of yes forecasts to the number of yes observations.
    # bias = 1 means Unbiased forecasts,indicating that the event was forecast the same number of times that it was observed
    # bias > 1 = indicates that the event was forecast more often than observed, which is called overforecasting
    #  bias < 1 = one indicates that the event was forecast less often than observed, or was underforecast

    # acc = round((a_hit + d_cor_neg)/ n,2)
    #This is simply the fraction of the n forecast occasions for which the nonprobabilistic forecast
    # correctly anticipated the subsequent event or non event

    # csi = round(a_hit/(a_hit + b_false + c_miss),2)
    # csi Range: 0 to 1, 0 indicates no skill. Perfect score: 1. It can be viewed as a proportion
    # correct for the quantity being forecast, after removing correct no forecasts from consideration.

    # hss = round((2*((a_hit*d_cor_neg) - (b_false*c_miss)))/(((a_hit+c_miss)*(c_miss+d_cor_neg))+((a_hit+b_false)*(b_false+d_cor_neg))),2)
    # hss Range: -1 to 1, 0 indicates no skill. Perfect score: 1.
    #round(((a_hit + d_cor_false)-(a_hit_rand_hss))/(N - a_hit_rand_hss),2)

    # ets = round((a_hit - a_ref)/(a_hit - a_ref + b_false + c_miss),2) 
    # ets Range: -1/3 to 1, 0 indicates no skill. Perfect score: 1.

    # Hanssen and Kuipers discriminant (true skill statistic, Peirce's skill score)
    #hk = round((a_hit/a_hit + b_miss) - (c_false/d_cor_false + c_false),3)

    return pod, far, bias, csi,  (a_hit, b_false, c_miss, d_cor_neg,n) #  pofd,hss, ets, acc, 


# CSI = a/a+b+c ; hit / hit + false + miss
# POD = a/a+c ; hit / hit + false
# FAR = b/a+b; false/ hit + false
# POFD = b/b+d; false/(false + correct_negative)
# Bias = a+b/a+c; (hit + false) / (hit + miss)
# aref =  (a+b)(a+c)/n.
# ETS = (a−aref)/ a−aref+b+c
# HSS = 2(ad−bc) / (a+c)(c+d)+ (a+b)(b+d )

#***********************************************************************************************************************************************************

# quantitative stats
# Note: always - model is simulated and obs is observed 
# bias ratio
def bias_ratio(obs,model):
    mu_obs = np.nanmean(obs)
    mu_model = np.nanmean(model)

    return mu_model/mu_obs
#***********************************************************************************************************************************************************

# relative bias
def relative_bias(obs,model):
    mu_residuals = np.nanmean(model - obs)
    mu_obs = np.nanmean(obs)

    return mu_residuals/mu_obs
#***********************************************************************************************************************************************************

# rmse
# Range 0 RMSE < inf, smaller is better.
# Notes: The standard deviation of the residuals. A lower spread indicates that the points are better concentrated
# around the line of best fit (linear). Random errors do not cancel. This metric will highlights larger errors.

def rmsqe(obs,model):
    return he.rmse(model,obs)

#***********************************************************************************************************************************************************

# mean normaised rmse (NRMSE)
# Compute the mean normalized root mean square error between the simulated and observed data.
# Range 0 NRMSE < inf, smaller is better.
# Notes: This metric is the RMSE normalized by the mean of the observed time series (x). Normalizing allows
# comparison between data sets with different scales.

def nrmsqe(obs,model):
    return he.nrmse_mean(model,obs)

#***********************************************************************************************************************************************************
# pearson correlation

# Range: -1 R (Pearson) 1. 1 indicates perfect postive correlation, 0 indicates complete randomness, -1 indicate
# perfect negative correlation.
# Notes: The pearson r coefficient measures linear correlation. It is sensitive to outliers.

def p_corr(obs,model):
    return he.pearson_r(model,obs)

#***********************************************************************************************************************************************************

# coefficient of determination

# Range: 0 r2 1. 1 indicates perfect correlation, 0 indicates complete randomness.

# Notes: The Coefficient of Determination measures the linear relation between simulated and observed data.
# Because it is the pearson correlation coefficient squared, it is more heavily affected by outliers than the pearson
# correlation coefficient.
def r_sqr(obs,model):
    return he.r_squared(model,obs)
#***********************************************************************************************************************************************************

# Nash-Sutcliffe Efficiency

# Range: -inf < NSE < 1, does not indicate bias, larger is better.

# Notes: The Nash-Sutcliffe efficiency metric compares prediction values to naive predictions (i.e. average value).
# One major flaw of this metric is that it punishes a higher variance in the observed values (denominator). This
# metric is analogous to the mean absolute error skill score (MAESS) using the mean flow as a benchmark.

# proposed by Nash and Sutcliffe (1970) is defined as one minus the sum of the absolute squared differences
# between the predicted and observed values normalized by the variance of the observed values during the period under
# investigation.
def n_se(obs,model):
    return he.nse(model,obs)

#***********************************************************************************************************************************************************

#Kling-Gupta efficiency (2009).

# Range: -inf < KGE (2009) < 1, larger is better.

# Notes: Gupta et al. (2009) created this metric to demonstrate the relative importance of the three components
# of the NSE, which are correlation, bias and variability. This was done with hydrologic modeling as the context.
# This metric is meant to address issues with the NSE.
def kge2009(obs,model):
    return he.kge_2009(model,obs)


#***********************************************************************************************************************************************************

#Kling-Gupta efficiency (2012).

# Range: -inf < KGE (2009) < 1, larger is better.

# Notes: The modified version of the KGE (2009). Kling proposed this version to avoid cross-correlation between
# bias and variability ratios.
def kge2012(obs,model,how):
    if how == 'all':
        return he.kge_2012(model,obs,return_all=True)
    elif how == 'kge':
        return he.kge_2012(model,obs,return_all=False)


#***********************************************************************************************************************************************************
# Compute the the mean absolute percentage error (MAPE).
# Range: 0% MAPE inf. 0% indicates perfect correlation, a larger error indicates a larger percent error in the
# data.
def mape(obs,model):
    he.mape(model,obs,remove_neg=True)
