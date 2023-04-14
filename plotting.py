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
from spynal import info, plots


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
        
def plot_avg_wave(allAlignWaves, df, labels, comp_num, area):
    fig, axs = plt.subplots(comp_num, 1)
    fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(10)
    
    clusters =[]
    
    colors = ['xkcd:azure', 'mediumseagreen', 'tab:olive', 'xkcd:lavender', 'b', 'g']
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
    plt.figure(figsize = (10, 10))
    datapts, features = df.shape
    labels = []
    percs = []
    for i in range(comp_num):
        cluster_pts = df.loc[df['labels'] == i].shape[0]
        perc = cluster_pts/datapts
        percs.append(perc)
        labels.append(str(i))
    
    plt.pie(percs, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title(area + " datapoints per cluster")
    plt.show()
    
def var_values(df, all_params, comp_num, area):
    fig, axs = plt.subplots(1, len(all_params))
    fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    colors = ['xkcd:azure', 'mediumseagreen', 'tab:olive', 'xkcd:lavender', 'b', 'g']
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
        
    plt.show()
    
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
    
def pev_plot(data, labels, p, area, label_type):
    plt.figure()
    comp_num = max(labels)
    #fig, axs = plt.subplots(comp_num, 1)
    #fig.tight_layout()
    colors = ['xkcd:azure', 'mediumseagreen', 'tab:olive', 'xkcd:lavender']
    clusters = []
    time_vec = np.linspace(-1, 1.95, 60)
    
    for i in range(comp_num):
        #ax = axs[i]
        clusters.append(i)
        cluster_units = labels == i
        
        data_mean = np.mean(data[cluster_units, :], axis = 0)
        plt.plot(time_vec, data_mean, color=colors[i])
        
        sems = stats.sem(data)
        plt.fill_between(time_vec, data_mean-sems, data_mean+sems,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    for i in range(len(p)):
        if p[i] <= 0.05:
            plt.plot(time_vec[i], p[i], marker='o', markersize=10, markerfacecolor='red')
    plt.title(area + " pev plot for " + label_type, y = 1.03, fontsize = 20)
    #fig.legend(clusters)

def raster():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
    