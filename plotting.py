#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:47:31 2023

@author: ehua
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy import stats
import pandas as pd
from sklearn.manifold import TSNE
from spynal import spikes
import math

def outlier_id(labels_df, comp_num):
    rows, _ = labels_df.shape
    max_std = 2.5
    
    outlier_col = np.empty((rows, ))
    for i in range(comp_num):
        cluster_units = labels_df.loc[labels_df['labels'] == i]
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
    
    labels_df['outliers'] = outlier_col
    return labels_df

def pairplot(labels_df, outliers_df, comp_num, area, outlier=False):
    all_params = ['troughToPeak', 'repolTime', 'meanRates', 'CV', 'LV']
    if outlier:
        for i in range(comp_num):
            cluster_units = outliers_df.loc[outliers_df['labels'] == i]
            g = sns.pairplot(cluster_units, hue = "outliers", kind='scatter', 
                                    diag_kind='kde', palette = 'muted',  x_vars = all_params, y_vars = all_params)
            g.fig.suptitle(area + " outlier pairplot for comp " + i, y = 1.03, fontsize = 20)
    else:
        g = sns.pairplot(labels_df, hue = "labels", kind='scatter', 
                                diag_kind='kde', palette = 'muted',  x_vars = all_params, y_vars = all_params)
        g.fig.suptitle(area + " cluster pairplot", y = 1.03, fontsize = 20)
        plt.savefig('/home/ehua/clustering/090623_data/figures/{}_pairplot.png'.format(area))

        
def plot_avg_wave(allAlignWaves, df, comp_num, area):
    fig, axs = plt.subplots(comp_num, 1)
    fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(10)
    
    clusters =[]
    
    colors = sns.color_palette("muted")
    for i in range(comp_num):
        ax = axs[i]
        clusters.append(i)
        cluster_units = df.loc[df['labels'] == i]
        cluster_units_idx = cluster_units.index
        outlier_idx = cluster_units.loc[cluster_units["outliers"] == 1].index
        cluster_units_idx = list(set(cluster_units_idx) - set(outlier_idx))
        cluster_waves = allAlignWaves[:, cluster_units_idx]
        outlier_waves = allAlignWaves[:, outlier_idx]
        
        ax.plot(cluster_waves, color = colors[i], alpha = 0.2)
        mean_wave = np.mean(cluster_waves, axis = 1)  
        ax.plot(mean_wave, color = 'k')
        if outlier_waves.size != 0:
            ax.plot(outlier_waves, color="crimson", alpha = 0.5)
    fig.suptitle(area + " cluster waveforms", y = 1.03, fontsize = 20)
    fig.legend(clusters)

    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_avg_waves.png'.format(area))

    
def elbow_plot(components, data_mean, data_std):
    plt.figure(0)
    plt.plot(components, data_mean, 'k', color='#CC4F1B')
    plt.fill_between(components, data_mean-data_std, data_mean+data_std,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
def tsne_plot(ax, perplexity, df_tsne, area):
    ax.set_title(area + " for Perp=%d" % perplexity)
    sns.scatterplot(data=df_tsne, x='comp1', y='comp2', marker='o', hue=df_tsne.label.astype('category').cat.codes, ax = ax)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")
    
def area_dist(df, comp_num, area):
    colors = sns.color_palette("muted")
    plt.figure(figsize = (10, 10))
    datapts, _ = df.shape
    labels = []
    percs = []
    for i in range(comp_num):
        cluster_pts = df.loc[df['0'] == i].shape[0]
        perc = cluster_pts/datapts
        percs.append(perc)
        labels.append(str(i))
    
    plt.pie(percs, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    plt.title(area + " datapoints per cluster")
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_dist.png'.format(area))
    
def param_values(df, all_params, comp_num, area):
    fig, axs = plt.subplots(1, len(all_params))
    fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    colors = sns.color_palette("muted")
    clusters = []
   
    for i in range(len(all_params)):
        ax = axs[i]
        param = all_params[i]
        param_df = df[[param, 'labels']]
        for j in range(comp_num):
            clusters.append(j)
            cluster_units = param_df.loc[param_df['labels'] == j]
            mean_param = np.mean(cluster_units[param])
            sem_param = stats.sem(cluster_units[param])
            ax.errorbar(x=0, y=mean_param, yerr=sem_param, fmt ='o', color = colors[j])
            ax.set_title(param)
   
    plt.legend(clusters)
    plt.title(area + " parameter values")
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_param_values.png'.format(area))
    
def psth(df, comp_num, blockRates, trialRates):
    time_vec = np.linspace(-1, 1.95, 60)
    fig, axs = plt.subplots(comp_num, 1)
    fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(10)
    
    for i in range(comp_num):
        ax = axs[i]
        cluster_units = df.loc[df['labels'] == i]
        cluster_units_idx = cluster_units.index
        
        ax.plot(time_vec, np.mean(blockRates[cluster_units_idx, :], axis = 0), color="crimson", linewidth = 2)
        ax.plot(time_vec, np.mean(trialRates[cluster_units_idx, :], axis = 0), color="cyan", linewidth = 2)
    
    fig.legend(["block", "trial"])
    
def pev_plot(data, labels, comp_num, area, label_type):
    plt.figure()
    #fig, axs = plt.subplots(comp_num, 1)
    #fig.tight_layout()
    colors = sns.color_palette("muted")
    clusters = []
    time_vec = np.linspace(-1, 0.5, 30)
    
    for i in range(comp_num):
        #ax = axs[i]
        clusters.append(i)
        cluster_units = labels == i
        
        data_mean = np.mean(data[cluster_units.to_numpy().flatten(), :], axis = 0)
        plt.plot(time_vec, data_mean, color=colors[i])
        
        sems = stats.sem(data)
        plt.fill_between(time_vec, data_mean-sems, data_mean+sems,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    plt.title(area + ' PEV plot for ' + label_type)
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_PEV_{}.png'.format(area, label_type))

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


def raster(spike_data, labels, comp_num, cond_data, cond_type):
    '''
    Generates raster plots of spike data per cluster for different PEV labels.
        
        Input: spike_data (n_units, n_timepts): ndarray of spike timestamps
               labels (n_units, ): column of cluster labels
               comp_num: number of clusters
               cond_data (n_units, ): labels of conditions
               cond_type: condition type (samp or pred)

        Output: scatterplot of data in 2-d space.

    '''
    fig = plt.figure(constrained_layout=True,figsize=(10,10))
    subfigs = fig.subfigures(math.ceil(comp_num/2),2)

    if cond_type == 'samp': # 3 samp types vs 2 pred types (block vs. trial)
        num_cond = 3
    else:
        num_cond = 2


    for i in range(comp_num):
        axs_i = subfigs[i].subplots(num_cond, 1, sharex=True)

        cluster_units = labels == i
        cluster_units_idx = cluster_units.index

        spike_units = spike_data[cluster_units_idx, :]

        for ax in axs_i:   
            cond_i = cond_data.loc[cond_data['0'] == i]
            cond_i_idx = cond_i.index
            cond_spikes = spike_units[cond_i_idx, :]                         
            
            spikes.plot_raster(cond_spikes, ax)
        
        ax.title('cluster ' + comp_num + ' raster for ' + cond_type)
    

def depth_analysis(depths, labels):
    # generate some vector of distinct depths
    # plot # of cluster of that depth in stacked bars..?
    pass


def main():
    areas = ['LIP']
    pev_types = ['samp', 'pred']

    for area in areas:
        #feat_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_df.csv'.format(area), index_col = 0)
        waves_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_waves.csv'.format(area), index_col = 0)
        spikes_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_spikes.csv'.format(area), index_col = 0)
        spikes = spikes_df.to_numpy()
        waves = waves_df.to_numpy()
        waves_ptp = waves.ptp(axis = 0)
        waves_norm = np.divide(waves, waves_ptp)
        
        all_params = ['troughToPeak', 'repolTime', 'meanRates', 'CV', 'LV']
        labels_df = pd.read_csv('/home/ehua/clustering/090623_data/clusters/{}_labels_df.csv'.format(area), index_col = 0)
        labels = pd.read_csv('/home/ehua/clustering/090623_data/clusters/{}_labels.csv'.format(area), index_col = 0)
        comp_num = max(labels['0']+1)

        cond_data = pd.read_csv('/home/ehua/clustering/090623_data/clusters/{}_{}.csv'.format(area, cond_type), index_col = 0)
        
        #feat_reduction(feat_df, labels, area)
        
        # outliers_df = outlier_id(labels_df, comp_num)
        # plot_avg_wave(waves_norm, outliers_df, comp_num, area)
        # pairplot(labels_df, outliers_df, comp_num, area)
        
        # area_dist(labels, comp_num, area)
        # param_values(labels_df, all_params, comp_num, area)

        #psth(labels_df, comp_num, allBlockRates, allTrialRates)
        
        # for pev_type in pev_types:
        #     pev_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_PEV_{}.csv'.format(area, pev_type), index_col = 0)
        #     pev_data = pev_df.to_numpy()
            
        #     pev_plot(pev_data, labels, comp_num, area, pev_type)

        for cond_type in pev_types:
            raster(spikes, labels, comp_num, cond_data, cond_type)

if __name__ == "__main__":
    main()
    