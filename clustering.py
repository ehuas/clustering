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
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
from pylab import *
import pandas as pd



def GMM(features, num_reps):
    components = np.arange(2, 10) # 2-9 clusters
    bics = np.zeros((num_reps, len(components)))
    aics = np.zeros((num_reps, len(components)))
    min_comps = []
    for rep in range(num_reps): # what is num_reps
        min_bic = np.inf 
        min_aic = np.inf
        for comp in components: # for each cluster #
            gmm = GaussianMixture(comp)
            gmm_copy = deepcopy(gmm)
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
    
    
    #plt.figure(0)
    bics_mean = np.mean(bics, axis=0)
    bics_stds = np.std(bics, axis = 0)
    #plt.plot(components, bics_mean, 'k', color='#CC4F1B')
    #plt.fill_between(components, bics_mean-bics_stds, bics_mean+bics_stds,
    #                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    average_min_comp = mode(min_comps)
    print(min_comps)
    print(average_min_comp)
    gmm_min = GaussianMixture(average_min_comp)
    gmm_min.fit(features)
    min_labels = gmm_min.predict(features)
    
    return gmm_min, min_labels, average_min_comp
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
    
    # average_min_comp = mode(min_comps)
    # gmm_min = GaussianMixture(average_min_comp)
    # gmm_min.fit(features)
    # min_labels = gmm_min.predict(features)
    # return min_labels, average_min_comp
    
def outlier_id(df, labels):
    df['labels'] = labels
    max_std = 2.5
    
    troughToPeak = df['troughToPeak'].to_numpy()
    repolTime = df['repolTime'].to_numpy()
    ttp_std = np.std(troughToPeak)
    ttp_mean = np.mean(troughToPeak)
    ttp_std_filter = np.where(np.logical_or(troughToPeak < ttp_mean - (max_std*ttp_std), troughToPeak > ttp_mean + (max_std*ttp_std)), 1, 0)
    rpt_std = np.std(repolTime)
    rpt_mean = np.mean(repolTime)
    rpt_std_filter = np.where(np.logical_or(repolTime < rpt_mean - (max_std*rpt_std), repolTime > rpt_mean + (max_std*rpt_std)), 1, 0)


    outlier_col = np.logical_or(ttp_std_filter, rpt_std_filter)    
    outlier_idx = np.nonzero(outlier_col)
    
    df['outliers'] = outlier_col
    return df, outlier_idx
    
def pairplot(labels_df, outliers_df):
    sns.pairplot(labels_df, hue = "labels", kind='scatter', 
                            diag_kind='kde', palette = 'muted')
    sns.scatterplot(outliers_df=outliers_df[outliers_df["outliers"] == 1], x="x", y="y", color="crimson", label="outlier")
    
def hue_regplot(data, x, y, hue, palette=None, **kwargs):
    from matplotlib.cm import get_cmap
    
    regplots = []
    
    levels = data[hue].unique()
    
    if palette is None:
        default_colors = get_cmap('tab10')
        palette = {k: default_colors(i) for i, k in enumerate(levels)}
    
    for key in levels:
        regplots.append(
            sns.regplot(
                x=x,
                y=y,
                data=data[data[hue] == key],
                color=palette[key],
                **kwargs
            )
        )
    
    return regplots

def feat_reduction(df, min_labels, area):
    num_pts, num_vars = df.shape
    print(num_pts)
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
    
        ax.set_title(area + " for Perp=%d" % perplexity)
        sns.scatterplot(data=df_tsne, x='comp1', y='comp2', marker='o', hue=df_tsne.label.astype('category').cat.codes, ax = ax)
        #hue_regplot(x='comp1', y='comp2', data=df_tsne, hue='label', ax=ax, fit_reg=False, marker='o')
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")
        
    # allDF_embedded = TSNE(n_components=2, learning_rate='auto',
    #                   init='random', perplexity=50).fit_transform(allDF)
    
def plot_avg_wave(allAlignWaves, cluster_stats_df, cluster_labels, comp_num):
    f = plt.figure()
    f.set_figheight(20)
    colors = ['c', 'm', 'g', 'y', 'b', 'r']
    for i in range(comp_num):
        cluster_units = cluster_stats_df.loc[cluster_stats_df['labels'] == i]
        cluster_units_idx = cluster_units.index
        cluster_waves = allAlignWaves[:, cluster_units_idx]
        plt.subplot(comp_num, 1, i+1)
        plt.plot(cluster_waves, color = colors[i], alpha = 0.2)
        mean_wave = np.mean(cluster_waves, axis = 1)  
        plt.plot(mean_wave, color = 'k')
    plt.show()


def main():    
    area = '7A'
    feat_df = pd.read_csv('/home/ehua/clustering/7A_df.csv', index_col = 0)
    allAlignWaves_df = pd.read_csv('/home/ehua/clustering/allAlignWaves_7A.csv', index_col = 0)
    allAlignWaves = allAlignWaves_df.to_numpy()
    waves_ptp = allAlignWaves.ptp(axis = 0)
    allAlignWaves_norm = np.divide(allAlignWaves, waves_ptp)
    
    all_params = ['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV']
   
    cluster_stats = feat_df[all_params].to_numpy()
    scaler = StandardScaler() 
    cluster_stats_norm = scaler.fit_transform(cluster_stats)
    
    gmm_min, min_labels, comp_num = GMM(cluster_stats_norm, 40)
    
    cluster_stats_df = pd.DataFrame(cluster_stats_norm)
    cluster_stats_df['labels'] = min_labels
    labels_df = feat_df.copy(deep=True)
    labels_df['labels'] = min_labels
    #plot_avg_wave(allAlignWaves_norm, cluster_stats_df, min_labels, comp_num)
    
    #feat_reduction(feat_df, min_labels, area)
    
    outliers_df, outlier_idx = outlier_id(feat_df, min_labels)
    pairplot(labels_df, outliers_df)

if __name__ == "__main__":
    main()
    