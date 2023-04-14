#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:52:30 2022

@author: huange
"""
from spynal import spikes, utils
from spynal.matIO import loadmat
import numpy as np
import pandas as pd
from utils import *
import copy

def trialSelection(trial_info, session_info):
    '''
    Selects valid trials from data. 
        
        Input: trialInfo (n_trials, n_variables) DataFrame for single session
        Output: (valid_trials,) vector of indices of units to keep
    
    Valid trials are defined as: 
        (a) correct and 
        (b) ≤ 5 trials before drug injection onset

    '''
    drugStartTrial = session_info['drugStartTrial']
    beforeDrugTrials = trial_info.loc[trial_info['trial'] <= drugStartTrial-5]
    trials_df = beforeDrugTrials.loc[beforeDrugTrials['correct']]
    trials_keep = np.where(beforeDrugTrials['correct'])[0]
    sampInfo = copy.deepcopy(trials_df['sample'])
    
    block_trials = np.where(trials_df['blockType'] == 'block')[0]
    trial_trials = np.where(trials_df['blockType'] == 'trial')[0]

    trials_df.loc[trials_df['blockType'] == 'block'] = 1
    trials_df.loc[trials_df['blockType'] == 'trial'] = 0
    predInfo = trials_df['blockType']  
    
    return trials_keep, predInfo, sampInfo, block_trials, trial_trials

def trialsKeep(trials_keep, spike_times, spike_waves):
    times_trials = spike_times[trials_keep, :]
    waves_trials = spike_waves[trials_keep, :]
    
    return times_trials, waves_trials
    
def neuronSelection(times_trials):
    '''
    Selects valid neurons from data. 
        
        Input: spikeTimes (n_trials, n_units) object array for single session
        Output: (n_units,) bool vector indicating which units to keep
    
    Valid neurons are defined as: 
        (a) having an overall mean spike rate > 1 spike/s (weeding out silent cells) and 
        (b) having < 0.1% ISIs within 1 ms (weeding out poorly isolated single neurons)

    '''    
    rates, timepts = spikes.rate(times_trials, method='bin', lims = [-1, 2])
    meanRates = np.mean(np.mean(rates, axis = 2), axis = 0) #takes mean over all trials & timepts
    
    neurons_keep = np.where(meanRates > 1)[0] #find all indices where mean rate > 1
    valid_spikes = times_trials[:, neurons_keep]
    
    allISIs = spikes.isi(valid_spikes)
    ms_ISI = np.multiply(allISIs, 1000)
    concat = utils.concatenate_object_array(ms_ISI, 0)
    if concat.size > 1:
        flatAll = np.squeeze(utils.concatenate_object_array(ms_ISI, 0)) #every ISI per unit
    else:
        flatAll = concat[0]
    
    for idx, neuron in enumerate(neurons_keep):
        shorts = np.where(flatAll[idx] <= 1)[0] #all ISIs less than 1 ms for neuron
        num_shorts = np.size(shorts)
        total_isi = np.size(flatAll[idx])
        if num_shorts/total_isi >= 0.1:
            neurons_keep = np.delete(neurons_keep, idx) #don't keep the neuron           
    
    return neurons_keep 


def neuronsKeep(neurons_keep, times_trials, waves_trials):
    times_data = times_trials[:, neurons_keep]
    waves_data = waves_trials[:, neurons_keep]
    
    return times_data, waves_data

def rateData(time_data):
    rates, timepts = spikes.rate(time_data, method='bin', lims = [-1, 2])
    meanRates = np.mean(np.mean(rates, axis = 2), axis = 0)
    meanRates = anscombe(np.expand_dims(meanRates, axis=0))
    
    return meanRates, rates

def isiData(time_data):
    allISIs = spikes.isi(time_data)
    ms_ISI = np.multiply(allISIs, 1000)
    
    return ms_ISI

def waveAlign(waves_data, spike_waves_schema, trial_subset_indices = None):
    '''
    Gets the mean-aligned waveform from data. 
        
        Input: spikeWaves (n_trials, n_units) object array for single session
               spikeWavesSchema: 
               trials_keep: all valid trials
               neurons_keep: all valid units
               trial_subset_indices: optional subset of (valid) trials of interest
               
        Output: meanAlignWaves (n_trials, n_units): aligned mean waveforms for each unit
                smpRate: 10x interpolated sampling rate
                
    '''
    
    if trial_subset_indices: #if we pass in some subset
        waves_data = waves_data[trial_subset_indices, :]
    
    n_trials, n_units = np.shape(waves_data)
    timepts = spike_waves_schema['elemIndex'][0]
    
    num_timepts = (np.size(timepts)-1)*10
    meanAlignWaves = np.zeros((num_timepts, n_units))
    
    for neuron in range(n_units):
        spikesAll = utils.concatenate_object_array(waves_data[:, neuron], axis = 0, elem_axis = 1) #unexpected elem_axis
        n_timepts, n_spikes = np.shape(spikesAll) #get # of time pts
        x = np.arange(1, n_timepts+1)
        xinterp = np.arange(1, n_timepts, 0.1) #keep length, divide step by 10
        waves_interp = utils.interp1(x, spikesAll, xinterp, axis = 0)
        meanWave = np.mean(waves_interp, axis=1) #get mean waveform over all spikes
        meanTroughIdx = np.argmin(meanWave) #get mean trough idx
        
        for spike_idx in range(n_spikes):
            spike = waves_interp[:, spike_idx]
            spikeTroughIdx = np.argmin(spike)
            diff = spikeTroughIdx - meanTroughIdx
            newSpike = np.full(np.shape(spike), np.nan)
            if diff > 0: # if the spike's trough is shifted ahead of mean trough
                newSpike[:-diff] = spike[diff:] #move it back
            elif diff < 0:
                newSpike[abs(diff):] = spike[:diff]
            else:
                newSpike = spike
            waves_interp[:, spike_idx] = newSpike
            
        # new waves_interp with aligned spikes
        meanAlignWave = np.nanmean(waves_interp, axis=1) #take mean of all spikes
        meanAlignWaves[:, neuron] = meanAlignWave
        
    smpRate = spike_waves_schema['smpRate']*10
    if waves_data.size != 0:
        alignWaves = waves_interp
    else:
        alignWaves = np.zeros((num_timepts, n_units))

        
    return meanAlignWaves, smpRate, alignWaves

def LV(ISIs):
    n_trials, n_units = np.shape(ISIs)
    allLV = np.zeros((1, n_units))
    
    for neuron in range(n_units):
        neuronLV = np.zeros((n_trials,))
        for trial in range(n_trials): 
            if len(ISIs[trial, neuron]) <= 1: #if there are no ISIs to compare against each other
                neuronLV[trial] = float('NaN')
            else: 
                LV = spikes.isi_stats(ISIs[trial, neuron], stat='LV')
                neuronLV[trial] = LV
        meanLV = np.nanmean(neuronLV, axis=0)
        allLV[:, neuron] = meanLV
    return allLV
    
def preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema):
    trials_keep, predInfo, sampInfo, block_trials, trial_trials = trialSelection(trial_info, session_info)
    times_trials, waves_trials = trialsKeep(trials_keep, spike_times, spike_waves)
    
    neurons_keep = neuronSelection(times_trials)
    time_data, waves_data = neuronsKeep(neurons_keep, times_trials, waves_trials)
    
    meanRates, rates = rateData(time_data)
    meanNeuronRate = np.mean(rates, axis=0)
    blockRates = np.mean(rates[block_trials, :, :], axis = 0)
    trialRates = np.mean(rates[trial_trials, :, :], axis = 0)
    
    ISIs = isiData(time_data)
    
    meanAlignWaves, smpRate, alignWaves = waveAlign(waves_data, spike_waves_schema)
    
    return trials_keep, neurons_keep, meanRates, ISIs, meanAlignWaves, smpRate, rates, alignWaves, meanNeuronRate, blockRates, trialRates, predInfo, sampInfo

def featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates):    
    '''
    Extracts features of interest from data. 
        
        Input: meanRates (n_units): mean spike rates for each unit 
               ISIs (n_trials, n_units): ISIs for each trial and unit
               meanAlignWaves (n_timepts, n_units): mean aligned wave for each unit
               smpRate: 10x interpolated sampling rate
        Output: featuresDF: dataframe containing features of interest
                            meanRates, troughToPeak, repolTime, CV, LV

    '''
    
    troughToPeak = spikes.waveform_stats(meanAlignWaves, stat='width', smp_rate=smpRate) #axis is 0?
    repolTime = spikes.waveform_stats(meanAlignWaves, stat='repolarization', smp_rate=smpRate)
    
    mean_timepts = np.mean(rates, axis=2)
    CV = spikes.rate_stats(mean_timepts, stat='CV', axis=0) #deal with timepts
    allLV = LV(ISIs)
    
    features = {'meanRates': np.squeeze(meanRates).tolist(), 'troughToPeak': np.squeeze(troughToPeak).tolist(), 'repolTime': np.squeeze(repolTime.tolist()), 'CV': np.squeeze(CV).tolist(), 'LV': np.squeeze(allLV).tolist()}
    featuresDF = pd.DataFrame(data=features)
    
    return featuresDF

def main():     
    #validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates = preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema)
    #featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
    pass

if __name__ == "__main__":
    main()

