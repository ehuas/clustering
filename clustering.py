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
from copy import deepcopy
from sklearn.manifold import TSNE
import pandas as pd
from plotting import *
from analysis import *


def GMM(features, num_reps):
    '''
    Performs GMM clustering on data. 
        
        Input: features (n_features, n_datapts): matrix of feature values for each datapoint
               num_reps: number of times clustering is performed for a certain component value

        Output: gmm_min: model fitted using the best number of components
                min_labels: cluster assignments for each data point
                average_min_comp = number of components used for clustering

    '''
    components = np.arange(2, 10) # 2-9 clusters
    bics = np.zeros((num_reps, len(components)))
    models = np.empty((0, len(components)))
    labels = []
    min_comps = []
    
    for rep in range(num_reps): # what is num_reps
        min_bic = np.inf 
        rep_models = []
        rep_labels = []
        for comp in components: # for each cluster #
            gmm = GaussianMixture(n_components=comp, random_state=rep)
            gmm_copy = deepcopy(gmm)
            gmm_copy.fit(features)
            label = gmm_copy.predict(features)
            
            bic = gmm_copy.bic(features)
            
            bics[rep, comp-2] = bic
            rep_models.append(gmm)
            rep_labels.append(label)
            
            if bic < min_bic:
                min_bic = bic
                min_comp = comp
        min_comps.append(min_comp)
        models = np.append(models, np.array(rep_models).reshape((1, 8)), axis=0)
        labels.append(rep_labels)
    
    # plt.figure(0)
    # bics_mean = np.mean(bics, axis=0)
    # bics_stds = np.std(bics, axis = 0)
    # plt.plot(components, bics_mean, 'k', color='#CC4F1B')
    # plt.fill_between(components, bics_mean-bics_stds, bics_mean+bics_stds,
    #                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    average_min_comp = mode(min_comps)
    print(min_comps)
    min_model_idx = np.argmin(bics[:, average_min_comp-2])
    min_model = models[min_model_idx, average_min_comp-2]
    min_labels = labels[min_model_idx][average_min_comp-2]
    
    return min_model, min_labels, average_min_comp
        
def outlier_id(df, labels, comp_num, allAlignWaves):
    df['labels'] = labels
    rows, cols = df.shape
    max_std = 2.5
    
    outlier_col = np.empty((rows, ))
    for i in range(comp_num):
        cluster_units = df.loc[df['labels'] == i]
        cluster_idx = cluster_units.index
        
        troughToPeak = cluster_units['troughToPeak'].to_numpy()
        repolTime = cluster_units['repolTime'].to_numpy()
        ttp_std = np.std(troughToPeak)
        ttp_mean = np.mean(troughToPeak)
        ttp_std_filter = np.where(np.logical_or(troughToPeak < ttp_mean - (max_std*ttp_std), troughToPeak > ttp_mean + (max_std*ttp_std)), 1, 0)
        rpt_std = np.std(repolTime)
        rpt_mean = np.mean(repolTime)
        rpt_std_filter = np.where(np.logical_or(repolTime < rpt_mean - (max_std*rpt_std), repolTime > rpt_mean + (max_std*rpt_std)), 1, 0)

        outlier_all = np.logical_or(ttp_std_filter, rpt_std_filter)    
        
        outlier_col[cluster_idx] = outlier_all
    
    df['outliers'] = outlier_col
    return df
        

def feat_reduction(df, min_labels, area):
    '''
    Reduces N-d data to a 2-d feature space using TSNE method.
        
        Input: df (n_features, n_datapts): dataframe of features to be reduced
               min_labels: cluster assignments for each datapoint
               area: cortical area of data

        Output: scatterplot of data in 2-d space.

    '''
    num_pts, num_vars = df.shape
    perplexities = [10, 30, 40, 50, 60, 70, 80, 100]
    (fig, subplots) = plt.subplots(2, 4, figsize=(16, 8))
    axes = subplots.flatten()

    for i, perplexity in enumerate(perplexities):
        ax = axes[i]
    
        tsne = TSNE(
            n_components=2,
            init="random",
            perplexity=perplexity,
            learning_rate="auto",
            n_iter = 5000
        )
        df_embedded = tsne.fit_transform(df)
        
        df_tsne = pd.DataFrame(df_embedded, columns=['comp1', 'comp2'])
        df_tsne["label"] = min_labels
        
        tsne_plot(ax, perplexity, df_tsne, area)


def main():    
    area = 'PFC'
    label_type = 'samp'
    feat_df = pd.read_csv('/home/ehua/clustering/PFC_df.csv', index_col = 0)
    allAlignWavesDf = pd.read_csv('/home/ehua/clustering/allAlignWaves_PFC.csv', index_col = 0)
    allAlignWaves = allAlignWavesDf.to_numpy()
    waves_ptp = allAlignWaves.ptp(axis = 0)
    allAlignWaves_norm = np.divide(allAlignWaves, waves_ptp)
    
    #allRatesDf = pd.read_csv('/home/ehua/clustering/allRates_V4.csv', index_col = 0)
    #allRates = allRatesDf.to_numpy()
    
    #allBlockRatesDf = pd.read_csv('/home/ehua/clustering/allBlockRates_V4.csv', index_col = 0)
    #allBlockRates = allBlockRatesDf.to_numpy()
    
    #allTrialRatesDf = pd.read_csv('/home/ehua/clustering/allTrialRates_V4.csv', index_col = 0)
    #allTrialRates = allTrialRatesDf.to_numpy()
    
    all_params = ['troughToPeak', 'repolTime', 'meanRates', 'CV', 'LV']
   
    cluster_stats = feat_df[all_params].to_numpy()
    scaler = StandardScaler() 
    cluster_stats_norm = scaler.fit_transform(cluster_stats)
    
    gmm_min, min_labels, comp_num = GMM(cluster_stats_norm, 1000)
    
    cluster_stats_df = pd.DataFrame(cluster_stats_norm)
    cluster_stats_df['labels'] = min_labels
    labels_df = feat_df.copy(deep=True)
    labels_df['labels'] = min_labels
    
    feat_reduction(feat_df, min_labels, area)
    
    outliers_df = outlier_id(feat_df, min_labels, comp_num, allAlignWaves_norm)
    plot_avg_wave(allAlignWaves_norm, outliers_df, min_labels, comp_num, area)
    #pairplot(labels_df, outliers_df, comp_num, area)
    
    #area_dist(labels_df, comp_num, area)
    var_values(labels_df, all_params, comp_num, area)
    
    #psth(labels_df, comp_num, allBlockRates, allTrialRates)
    
    allPEVDf = pd.read_csv('/home/ehua/clustering/allPEV_samp_PFC.csv', index_col = 0)
    allPEV = allPEVDf.to_numpy()
    
    pev = pev_plot(allPEV, min_labels, comp_num, area, label_type)

if __name__ == "__main__":
    main()
    