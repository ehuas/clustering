#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:14:04 2022

@author: huange
"""

import os
os.chdir('/Users/huange/clustering')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from statistics import mode
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns

def GMM(features, num_reps):
    components = np.arange(2, 10) # 2-9 clusters
    bics = np.zeros((num_reps, len(components)))
    aics = np.zeros((num_reps, len(components)))
    min_comps = []
    plt.figure(0)
    for rep in range(num_reps): # what is num_reps
        min_bic = np.inf 
        min_aic = np.inf
        for comp in components: # for each cluster #
            gmm = GaussianMixture(comp)
            gmm_copy = copy.deepcopy(gmm)
            gmm_copy.fit(features)
            
            bic = gmm_copy.bic(features)
            aic = gmm_copy.aic(features)
            
            bics[rep, comp-2] = bic
            aics[rep, comp-2] = aic
            
            if bic < min_bic:
                min_bic = bic
                min_comp = comp
            if aic < min_aic: 
                min_aic = aic
        min_comps.append(min_comp)
        plt.plot(components, bics[rep, :], label=str(rep), marker="o")
        plt.title('all bics, 40 reps')
    
    
    # plt.figure(1)
    # plt.boxplot(bics)
    # plt.figure(2)
    # plt.hist(min_comps)
    average_min_comp = mode(min_comps)
    print(min_comps)
    print(average_min_comp)
    gmm_min = GaussianMixture(2)
    gmm_min.fit(features)
    min_labels = gmm_min.predict(features)
    
    return gmm_min, min_labels
    #plt.scatter(features[:, 0], features[:, 1], c=min_labels, s=40, cmap='viridis')
    # bics_mean = np.mean(bics, axis=0) #average per cluster (over reps)
    # aics_mean = np.mean(aics, axis=0)
    # plt.figure(3)
    # plt.plot(components, bics_mean)
    # plt.title('bics_mean')
    # plt.xlabel('cluster #')
    # plt.figure(4)
    # plt.plot(components, aics_mean)
    # plt.title('aics_mean')
    # plt.xlabel('cluster #')
    
# =============================================================================
#    
#     average_min_comp = mode(min_comps)
#     gmm_min = GaussianMixture(average_min_comp)
#     gmm_min.fit(features)
#     min_labels = gmm_min.predict(features)
#     return min_labels, average_min_comp
# =============================================================================

def main():    
    # area = '7A'
    # path = '/Volumes/common/scott/laminarPharm/mat/Lucky-DMS_ACSF_PFC-10162020-001.mat'
    # spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
    # area_spike_times, area_spike_waves = select_area(unit_info, spike_times, spike_waves, area)
    
    # validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates = preProcessing(area_spike_times, trial_info, session_info, area_spike_waves, spike_waves_schema)
    # featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
    
    all_params = ['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV']
    
    cluster_stats = featuresDF[all_params].to_numpy()
    scaler = StandardScaler() 
    cluster_stats_norm = scaler.fit_transform(cluster_stats)
    gmm_min, min_labels = GMM(cluster_stats_norm, 40)
    
    featuresDF['cluster_labels'] = min_labels
    sns.pairplot(featuresDF, hue = "cluster_labels", kind='scatter', 
                            diag_kind='kde')

if __name__ == "__main__":
    main()
    