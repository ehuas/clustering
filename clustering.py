#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:14:04 2022

@author: huange
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from statistics import mode
from sklearn.preprocessing import StandardScaler
import pandas as pd
from plotting import *
from analysis import *


def GMM(features, num_reps, area):
    '''
    Performs GMM clustering on data. 
        
        Input: features (n_features, n_datapts): matrix of feature values for each datapoint
               num_reps: number of times clustering is performed for a certain component value

        Output: gmm_min: model fitted using the best number of components
                min_labels: cluster assignments for each data point
                average_min_comp = number of components used for clustering

    '''
    components = np.arange(2, 27) # 2-9 clusters
    bics = np.zeros((num_reps, len(components)))
    aics = np.zeros((num_reps, len(components)))
    models = np.empty((0, len(components)))
    labels = []
    min_comps = []
    
    for rep in range(num_reps): # what is num_reps
        min_bic = np.inf 
        rep_models = []
        rep_labels = []
        for comp in components: # for each cluster #
            gmm = GaussianMixture(n_components=comp, random_state=rep)
            gmm.fit(features)
            label = gmm.predict(features)
            
            bic = gmm.bic(features)
            aic = gmm.aic(features)
            
            bics[rep, comp-2] = bic
            aics[rep, comp-2] = aic
            rep_models.append(gmm)
            rep_labels.append(label)
            
            if bic < min_bic:
                min_bic = bic
                min_comp = comp
        min_comps.append(min_comp)
        models = np.append(models, np.array(rep_models).reshape((1, 25)), axis=0)
        labels.append(rep_labels)

    plt.figure(0)
    bics_mean = np.mean(bics, axis=0)
    bics_stds = np.std(bics, axis = 0)
    plt.plot(components, bics_mean, 'k', color='#CC4F1B')
    plt.fill_between(components, bics_mean-bics_stds, bics_mean+bics_stds,
                    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title(area + " bics")
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_bics.png'.format(area))
    
    plt.figure(1)
    aics_mean = np.mean(aics, axis=0)
    aics_stds = np.std(aics, axis = 0)
    plt.plot(components, aics_mean, 'k', color='#CC4F1B')
    plt.fill_between(components, aics_mean-aics_stds, aics_mean+aics_stds,
                    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title(area + " aics")
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_aics.png'.format(area))
    
    average_min_comp = mode(min_comps)
    print(min_comps)
    min_model_idx = np.argmin(bics[:, average_min_comp-2])
    min_model = models[min_model_idx, average_min_comp-2]
    min_labels = labels[min_model_idx][average_min_comp-2]
    
    return min_model, min_labels, average_min_comp\

def main():    
    area = 'dlPFC'
    feat_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_df.csv'.format(area), index_col = 0)
    waves_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_waves.csv'.format(area), index_col = 0)
    depths = pd.read_csv('/home/ehua/clustering/090623_data/{}_depths.csv'.format(area), index_col = 0)
    waves = waves_df.to_numpy()
    waves_ptp = waves.ptp(axis = 0)
    waves_norm = np.divide(waves, waves_ptp)
    
    all_params = ['troughToPeak', 'repolTime', 'meanRates', 'CV', 'LV']
   
    cluster_stats = feat_df[all_params].to_numpy()
    scaler = StandardScaler() 
    cluster_stats_norm = scaler.fit_transform(cluster_stats)
    
    _, min_labels, _ = GMM(cluster_stats_norm, 500, area)
    
    cluster_stats_df = pd.DataFrame(cluster_stats_norm)
    cluster_stats_df['labels'] = min_labels
    labels_df = feat_df.copy(deep=True)
    labels_df['labels'] = min_labels
    labels_df['depths'] = depths
    
    labels_df.to_csv('/home/ehua/clustering/090623_data/clusters/{}_labels_df.csv'.format(area))
    min_labels_df = pd.DataFrame(min_labels)
    min_labels_df.to_csv('/home/ehua/clustering/090623_data/clusters/{}_labels.csv'.format(area))

if __name__ == "__main__":
    main()
    