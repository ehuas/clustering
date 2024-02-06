#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:47:31 2023

@author: ehua
"""
import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy import stats
import pandas as pd
from sklearn.manifold import TSNE
from spynal import spikes
import math
from copy import deepcopy
from analysis import cluster_count

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

def pairplot(labels_df, outliers_df, comp_num, area, c_palette, outlier=False):
    all_params = ['troughToPeak', 'repolTime', 'meanRates', 'CV', 'LV']
    if outlier:
        for i in range(comp_num):
            cluster_units = outliers_df.loc[outliers_df['labels'] == i]
            g = sns.pairplot(cluster_units, hue = "outliers", kind='scatter', 
                                    diag_kind='kde', palette = c_palette,  x_vars = all_params, y_vars = all_params)
            g.fig.suptitle(area + " outlier pairplot for comp " + i, y = 1.03, fontsize = 20)
    else:
        g = sns.pairplot(labels_df, hue = "labels", kind='scatter', 
                                diag_kind='kde', palette = c_palette,  x_vars = all_params, y_vars = all_params)
        g.fig.suptitle(area + " cluster pairplot", y = 1.03, fontsize = 20)
        plt.savefig('/home/ehua/clustering/090623_data/figures/{}_pairplot.png'.format(area))

    plt.close()

        
def plot_avg_wave(allAlignWaves, df, comp_num, area, c_palette):
    fig, axs = plt.subplots(math.ceil(comp_num/2), 2)
    fig.set_figheight(15)
    fig.set_figwidth(10)
    ax_count = 0
    
    handles = []
    labels = []

    for i in range(comp_num):
        r, c = divmod(ax_count, 2)
        ax = axs[r, c]
        cluster_units = df.loc[df['labels'] == i]
        cluster_units_idx = cluster_units.index
        outlier_idx = cluster_units.loc[cluster_units["outliers"] == 1].index
        cluster_units_idx = list(set(cluster_units_idx) - set(outlier_idx))
        cluster_waves = allAlignWaves[:, cluster_units_idx]
        outlier_waves = allAlignWaves[:, outlier_idx]
        
        ax.plot(cluster_waves, color = c_palette[i], alpha = 0.2)
        mean_wave = np.mean(cluster_waves, axis = 1)  
        ax.plot(mean_wave, color = 'k')
        if outlier_waves.size != 0:
            ax.plot(outlier_waves, color="darkgrey", alpha = 0.5)
        ax.set_xlabel('Timepoint (10x Interpolated)')
        ax.set_ylabel('Amplitude (mV)')

        handles.append(lines.Line2D([0], [0], ls = '-', c=c_palette[i]))
        labels.append('Cluster ' + str(i))

        ax_count += 1
    
    handles.append(lines.Line2D([0], [0], ls = '-', c='k'))
    handles.append(lines.Line2D([0], [0], ls = '-', c='darkgrey'))

    labels.append('Average Waveform')
    labels.append('Outlier Wave')

    fig.legend(handles, labels)
    fig.suptitle(area + " cluster waveforms", y = 1.03, fontsize = 20)
    fig.tight_layout()

    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_avg_waves.png'.format(area))
    plt.close()
    
def elbow_plot(components, data_mean, data_std):
    plt.figure(0)
    plt.plot(components, data_mean, 'k', color='#CC4F1B')
    plt.fill_between(components, data_mean-data_std, data_mean+data_std,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
def tsne_plot(ax, perplexity, df_tsne, area, palette):
    ax.set_title(area + " for Perp=%d" % perplexity)
    sns.scatterplot(data=df_tsne, x='comp1', y='comp2', marker='o', hue=df_tsne.label.astype('category').cat.codes, palette = palette, ax = ax)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")
    
def area_dist(df, comp_num, area, c_palette):
    plt.figure(figsize = (10, 10))
    datapts, _ = df.shape
    labels = []
    percs = []
    for i in range(comp_num):
        cluster_pts = df.loc[df['0'] == i].shape[0]
        perc = cluster_pts/datapts
        percs.append(perc)
        labels.append('Cluster ' + str(i))
    
    plt.pie(percs, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=c_palette)
    plt.title(area + " Clusteral Distribution of Datapoints")
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_dist.png'.format(area))

    plt.close()
    
def param_values(df, all_params, comp_num, area, c_palette):
    fig, axs = plt.subplots(1, len(all_params))
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.suptitle(area + " parameter values")
    
    clusters = []
   
    for i in range(len(all_params)):
        ax = axs[i]
        param = all_params[i]
        param_df = df[[param, 'labels']]
        for j in range(comp_num):
            clusters.append('Cluster ' + str(j))
            cluster_units = param_df.loc[param_df['labels'] == j]
            mean_param = np.mean(cluster_units[param])
            sem_param = stats.sem(cluster_units[param])
            ax.errorbar(x=0, y=mean_param, yerr=sem_param, fmt ='o', color = c_palette[j])
            ax.set_title(param)
   
    plt.legend(clusters, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_param_values.png'.format(area))

    plt.close()
    
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

    plt.close()
    
def pev_plot(data, labels, comp_num, area, label_type, c_palette):
    plt.figure()
    #fig, axs = plt.subplots(comp_num, 1)
    #fig.tight_layout()
    clusters = []
    time_vec = np.linspace(-1, 0.5, 30)
    
    for i in range(comp_num):
        #ax = axs[i]
        clusters.append('Cluster ' + str(i))
        cluster_units = labels == i
        
        data_mean = np.mean(data[cluster_units.to_numpy().flatten(), :], axis = 0)
        plt.plot(time_vec, data_mean, color=c_palette[i])
        
        sems = stats.sem(data)
        # plt.fill_between(time_vec, data_mean-sems, data_mean+sems,
        #              alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label="SEM")
    
    plt.title(area + ' PEV plot for ' + label_type)
    plt.legend(clusters, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Relative to Sample Onset (s)')
    plt.ylabel('Percent Explained Variance')
    plt.tight_layout()
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_PEV_{}.png'.format(area, label_type))
    plt.close()

def feat_reduction(df, min_labels, area, palette):
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
        
        tsne_plot(ax, perplexity, df_tsne, area, palette)

    fig.suptitle('TSNE at different perplexities for Area ' + area)
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_TSNE.png'.format(area))
    plt.close()

def raster(area, labels, cluster, cond_data, cond_type, unit_num, cluster_units):
    '''
    Generates raster plots of spike data per cluster for different PEV labels.
        
        Input: spike_data (n_units, n_timepts): ndarray of spike timestamps
               labels (n_units, ): column of cluster labels
               comp_num: number of clusters
               cond_data (n_units, ): labels of conditions
               cond_type: condition type (samp or pred)

        Output: scatterplot of data in 2-d space.

    '''
    fig, axs = plt.subplots(math.ceil(unit_num/10), 10, figsize=(30,30))
    fig.suptitle('Raster Plot of Spikes for ' + cond_type + ' in ' + area)

    cond_name = str(cond_type + 'Info')
    ax_count = 0

    for i in cluster_units.index:
        
        spikes_i = np.load('/home/ehua/clustering/090623_data/spikes/{}_spikes_{}.npy'.format(area, i), allow_pickle=True)
        info_i = pd.read_csv('/home/ehua/clustering/090623_data/info/{}_{}_{}.csv'.format(area, cond_name, i), index_col = 0)
        
        r, c = divmod(ax_count, 10)
        ax = axs[r, c]

        spikes_df = pd.DataFrame(spikes_i)
        info_i.index = spikes_df.index
        df = pd.concat([spikes_df, info_i], axis = 1)

        if cond_type == 'samp':
            df = df.sort_values('sample')
            df = df.reset_index(drop = True)

            spikes.plot_raster(df[0], ax = ax)

            last1Trial = df['sample'].where(df['sample']==1.0).last_valid_index()
            last2Trial = df['sample'].where(df['sample']==2.0).last_valid_index()
            last3Trial = df['sample'].where(df['sample']==3.0).last_valid_index()
            ax.axhspan(0, last1Trial, facecolor='b', alpha=0.3)
            ax.axhspan(last1Trial, last2Trial, facecolor='m', alpha=0.3)
            ax.axhspan(last2Trial, last3Trial, facecolor='y', alpha=0.3)
            #plt.axhline(y = last1Trial, color = 'r', linestyle = '-') 
            #plt.axhline(y = last2Trial, color = 'r', linestyle = '-') 

        else:
            df = df.sort_values('blockType')
            df = df.reset_index(drop = True)

            spikes.plot_raster(df[0], ax=ax)

            last1Trial = df['blockType'].where(df['blockType']==0).last_valid_index()
            last2Trial = df['blockType'].where(df['blockType']==1).last_valid_index()
            ax.axhspan(0, last1Trial, facecolor='b', alpha=0.3)
            ax.axhspan(last1Trial, last2Trial, facecolor='m', alpha=0.3)
    
        ax_count += 1

    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_{}_raster_{}.png'.format(area, cluster, cond_type))

    plt.close()

def depth_analysis(depths, labels, comp_num, area, c_palette, jitter = 'no_jitter'):
    depths = depths.rename(columns={"0": "Depth (relative to L4)"})
    labels = labels.rename(columns={"0": "Label"})
    df = pd.concat([depths, labels], axis=1)
    
    fig, ax = plt.subplots(figsize=(19,19))
    df.Label = df.Label.astype("category")
    sns.swarmplot(data=df, x="Label", y="Depth (relative to L4)", palette=c_palette, ax=ax)

    # weights = np.ones(len(labels))
    # for i in range(len(labels)):
    #     weights[i] = 1/counts[labels['label'][i]]

    # sns.histplot(data=df, x="depth", hue="label", multiple="stack", weights = weights, palette='muted')

    clusters = []
    for i in range(comp_num):
        clusters.append('Cluster ' + str(i))
    
    plt.title(area + ' Depth Distribution per Cluster')
    plt.legend(clusters, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('/home/ehua/clustering/090623_data/figures/{}_cluster_depth_{}.png'.format(area, jitter))

    plt.close()

def main():
    areas = ['7A', 'LIP', 'V4']
    pev_types = ['samp', 'pred', 'unpredSamp']
    c_palette = sns.color_palette('colorblind')

    for area in areas:
        feat_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_df.csv'.format(area), index_col = 0)
        waves_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_waves.csv'.format(area), index_col = 0)
        waves = waves_df.to_numpy()
        waves_ptp = waves.ptp(axis = 0)
        waves_norm = np.divide(waves, waves_ptp)

        jitter_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_depths_jitter.csv'.format(area), index_col = 0)
        jitter = jitter_df.to_numpy()
        depths_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_depths.csv'.format(area), index_col = 0)
        depths = depths_df.to_numpy()

        
        all_params = ['troughToPeak', 'repolTime', 'meanRates', 'CV', 'LV']
        labels_df = pd.read_csv('/home/ehua/clustering/090623_data/clusters/{}_labels_df.csv'.format(area), index_col = 0)
        labels = pd.read_csv('/home/ehua/clustering/090623_data/clusters/{}_labels.csv'.format(area), index_col = 0)
        comp_num = max(labels['0']+1)
        counts = cluster_count(labels, comp_num)
        
        # depth_analysis(depths_df, labels, comp_num, area, c_palette)
        # depth_analysis(jitter_df, labels, comp_num, area, c_palette, jitter = 'jitter')
        
        #feat_reduction(feat_df, labels, area, c_palette)
        
        outliers_df = outlier_id(labels_df, comp_num)
        plot_avg_wave(waves_norm, outliers_df, comp_num, area, c_palette)
        # pairplot(labels_df, outliers_df, comp_num, area, c_palette)
        
        # area_dist(labels, comp_num, area, c_palette)
        # param_values(labels_df, all_params, comp_num, area, c_palette)

        #psth(labels_df, comp_num, allBlockRates, allTrialRates)
        
        # for pev_type in pev_types:
        #     pev_df = pd.read_csv('/home/ehua/clustering/090623_data/{}_PEV_{}.csv'.format(area, pev_type), index_col = 0)
        #     pev_data = pev_df.to_numpy()
            
        #     pev_plot(pev_data, labels, comp_num, area, pev_type, c_palette)

        # for cond_type in pev_types:
        #     # for cluster in range(comp_num):
        #         cluster = 6
        #         cond_data = pd.read_csv('/home/ehua/clustering/090623_data/{}_PEV_{}.csv'.format(area, cond_type), index_col = 0)

        #         cluster_units = labels.loc[labels['0'] == cluster]
        #         unit_num = cluster_units.size
        #         raster(area, labels, cluster, cond_data, cond_type, unit_num, cluster_units)

if __name__ == "__main__":
    main()
    