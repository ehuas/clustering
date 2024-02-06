#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:49:11 2023

@author: ehua
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 22:52:30 2022

@author: huange
"""
from spynal import spikes, utils, sync, spectra
import numpy as np
import pandas as pd
from utils import *
import copy
import math 
import matplotlib.pyplot as plt

def trialSelection(trial_info, session_info):
    '''
    Selects valid trials from data. 
        
        Input: trialInfo (n_trials, n_variables) DataFrame for single session
        Output: (valid_trials,) vector of indices of trials to keep
    
    Valid trials are defined as: 
        (a) correct and 
        (b) ≤ 5 trials before drug injection onset

    '''
    ### isolates only trials that are ≤ 5 trials before drug onset
    if 'drugStartTrial' in session_info:
        drugStartTrial = session_info['drugStartTrial']
        beforeDrugTrials = trial_info.loc[trial_info['trial'] <= drugStartTrial-5]
        trials_df = beforeDrugTrials.loc[beforeDrugTrials['correct']]
        trials_keep = np.where(beforeDrugTrials['correct'])[0]
    else:
        trials_keep = np.where(trial_info['correct'])[0]
        trials_df = trial_info.loc[trial_info['correct']]

    sampInfo = copy.deepcopy(trials_df['sample'])
    
    ### identifies the non predictable and predictable trials
    block_trials = np.where(trials_df['blockType'] == 'block')[0]
    trial_trials = np.where(trials_df['blockType'] == 'trial')[0]

    ### sets block types to 1, trial types to 0
    trials_df.loc[trials_df['blockType'] == 'block'] = 1 
    trials_df.loc[trials_df['blockType'] == 'trial'] = 0
    predInfo = copy.deepcopy(trials_df['blockType'])
    unpredSampInfo = sampInfo[trial_trials]

    return trials_keep, predInfo, sampInfo, block_trials, trial_trials, unpredSampInfo

def trialsKeep(trials_keep, spike_times, spike_waves):
    '''
    Filters out non-valid trials in given time, wave data
    '''
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
    rates, timepts = spikes.rate(times_trials, method='bin', lims = [-1, 0.5])
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
    '''
    Filters out non-valid units in given time, wave data
    '''
    times_data = times_trials[:, neurons_keep]
    waves_data = waves_trials[:, neurons_keep]
    
    return times_data, waves_data

def depth(neurons_keep, unit_info, jitter = True):
    '''
    Saves depth information for two dataests. Depth data is discretized -- jitter can be added to depths for ease of plotting.

    *Scaled Andre's data to match Alex's data 
    '''
    try:
        depths = unit_info['laminarDepth'][neurons_keep]
    except:
        ### Scale Andre's data
        depths = (unit_info['betaGammaDepth'][neurons_keep])/1000
    
    ### Create random vector of jitter
    x = pd.Series(np.random.uniform(low = -0.02, high = 0.02, size = len(depths)))

    ### Add jitter to original depths. Fill nan values with -3.
    jitter_depths = x.add(depths.reset_index(drop=True), fill_value = -3)
    
    depths = depths.fillna(-3)
    
    return depths, jitter_depths

def rateData(time_data):
    '''
    Returns:
        (a) meanRates: mean spike rate / unit
        (b) rates: full np array of spike rates (units x timepts x trials)
    '''

    rates, _ = spikes.rate(time_data, method='bin', lims = [-1, 0.5])
    meanRates = np.mean(np.mean(rates, axis = 2), axis = 0)
    meanRates = anscombe(np.expand_dims(meanRates, axis=0))
    
    return meanRates, rates

def isiData(time_data):
    '''
    Returns: ISI data from given spike data. On the scale of milliseconds (scaled by 1000).
    '''

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
        spikesAll = utils.concatenate_object_array(waves_data[:, neuron], axis = 0, elem_axis = 1) 
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

    return meanAlignWaves, smpRate

def LV(ISIs):
    '''
    Returns: 
    '''

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

def waveform_check(repolTime):
    '''
    Returns: passed_neurons (n_units, ): all units with non-inverted waveforms.
    '''

    passed_neurons = []
    row, num_neurons = np.shape(repolTime)
    for i in range(num_neurons): 
        if not math.isnan(repolTime[:, i]):
            passed_neurons.append(i)
    
    newRepolTime = repolTime[:, passed_neurons]
    return passed_neurons
            
def spike_i(spikes, predInfo, sampInfo, area, idx):
    '''
    Saves the time data for each unit (data is 1 x trials)
    '''
    trials, units = np.shape(spikes)
    for i in range(units):
        np.save('/home/ehua/clustering/090623_data/spikes/{}_spikes_{}.npy'.format(area, i+idx), spikes[:, i])

        predInfo.to_csv('/home/ehua/clustering/090623_data/info/{}_predInfo_{}.csv'.format(area, i+idx))
        sampInfo.to_csv('/home/ehua/clustering/090623_data/info/{}_sampInfo_{}.csv'.format(area, i+idx))

def filterSingleElectrodes(electrode_info, depths, lfp, area_idx, lfp_probe_idx):
    '''
    Returns: indices of non-singular electrodes
    '''
    idx_keep = np.where(electrode_info['elecType'][area_idx][lfp_probe_idx] != 'single')[0]
    depths = depths[idx_keep]
    lfp = lfp[:, idx_keep, :]

    return depths, lfp

def preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema, unit_info, area, unit_count):
    ### trial selection
    trials_keep, predInfo, sampInfo, block_trials, trial_trials, unpredSampInfo = trialSelection(trial_info, session_info)
    times_trials, waves_trials = trialsKeep(trials_keep, spike_times, spike_waves)
    
    ### neuron selection
    neurons_keep = neuronSelection(times_trials)
    time_data, waves_data = neuronsKeep(neurons_keep, times_trials, waves_trials)
    
    meanAlignWaves, smpRate = waveAlign(waves_data, spike_waves_schema)
    repolTime = spikes.waveform_stats(meanAlignWaves, stat='repolarization', smp_rate=smpRate)
    passed_neurons = waveform_check(repolTime)
    
    time_data = time_data[:, passed_neurons]

    # if time_data.shape[1] >= 2:
    #     spike_i(time_data, predInfo, sampInfo, area, unit_count)

    waves_data = waves_data[:, passed_neurons]
    meanAlignWaves = meanAlignWaves[:, passed_neurons]

    depths, jitter_depths = depth(passed_neurons, unit_info)
    
    meanRates, rates = rateData(time_data)
    meanNeuronRate = np.mean(rates, axis=0)
    blockRates = np.mean(rates[block_trials, :, :], axis = 0)
    trialRates = np.mean(rates[trial_trials, :, :], axis = 0)
    
    ISIs = isiData(time_data)
    
    return trials_keep, passed_neurons, meanRates, ISIs, meanAlignWaves, smpRate, rates, meanNeuronRate, blockRates, trialRates, predInfo, sampInfo, depths, jitter_depths, unpredSampInfo, trial_trials

def coupling(area_lfp, area_idx, depth_var, electrode_info, unit_info, spike_times, area, session, smp_rate):
    probeIDs = electrode_info['probeID'][area_idx].unique()

    for probeID in probeIDs:
        lfp_probe_idx = np.where(electrode_info['probeID'][area_idx] == probeID)[0]
        spk_probe_idx = np.where(unit_info['probeID'][area_idx] == probeID)[0]

        depths = electrode_info[depth_var][area_idx][lfp_probe_idx].to_numpy()

        lfp = area_lfp[:, lfp_probe_idx, :]
        spk = spike_times[:, spk_probe_idx]

        if depth_var == 'betaGammaDepth': 
            depths, lfp = filterSingleElectrodes(electrode_info, depths, lfp, area_idx, lfp_probe_idx)
        
        ### get idx of depth - superficial is negative, deep is positive. labels of 0 (layer 4) ignored
        sup_idx = np.where(depths < 0)[0]
        deep_idx = np.where(depths > 0)[0]

        lfp_sup = np.squeeze(np.mean(lfp[:, sup_idx, :], axis = 1))
        lfp_deep = np.squeeze(np.mean(lfp[:, deep_idx, :], axis = 1))

        spike_trains = spikes.times_to_bool(spk, lims=(-1,2))[0]

        _, n_units, _ = spike_trains.shape

        for unit in range(n_units):
            unit_spikes = np.transpose(np.squeeze(spike_trains[:, unit, :]))
            osc_sup,freqs_sup,timepts_sup,n_sup, phi_sup = \
                sync.spike_field_coupling(np.transpose(unit_spikes), 
                                        np.transpose(lfp_sup), 
                                        time_axis = 1, 
                                        smp_rate = smp_rate, 
                                        return_phase = True) \
                                        
            osc_deep,freqs_deep,timepts_deep,n_deep, phi_deep = \
                sync.spike_field_coupling(np.transpose(unit_spikes), 
                                        np.transpose(lfp_deep), 
                                        time_axis = 1, 
                                        smp_rate = smp_rate, 
                                        return_phase = True) \
            

            np.save('/home/ehua/clustering/090623_data/osc/{}_osc_sup_{}_{}_{}'.format(area, session, probeID, unit), np.squeeze(osc_sup))
            np.save('/home/ehua/clustering/090623_data/osc/{}_osc_deep_{}_{}_{}'.format(area, session, probeID, unit), np.squeeze(osc_deep))

            np.save('/home/ehua/clustering/090623_data/osc/{}_phi_sup_{}_{}_{}'.format(area, session, probeID, unit), phi_sup)
            np.save('/home/ehua/clustering/090623_data/osc/{}_phi_deep_{}_{}_{}'.format(area, session, probeID, unit), phi_deep)

            sup_path = '/home/ehua/clustering/090623_data/figures/{}_spec_sup_{}_{}_{}'.format(area, session, probeID, unit)
            deep_path = '/home/ehua/clustering/090623_data/figures/{}_spec_deep_{}_{}_{}'.format(area, session, probeID, unit)

            sup_img, sup_ax = spectra.plot_spectrogram(timepts_sup, freqs_sup, np.squeeze(osc_sup), sup_path, area, session, probeID, unit)
            deep_img, deep_ax = spectra.plot_spectrogram(timepts_deep, freqs_deep, np.squeeze(osc_deep), deep_path, area, session, probeID, unit)

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
    if np.shape(troughToPeak) == (1, 1):
        featuresDF = pd.DataFrame(data=features, index = [0])
    else:
        featuresDF = pd.DataFrame(data=features)
    
    return featuresDF

def main():     
    #validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates = preProcessing(spike_times, trial_info, session_info, spike_waves, spike_waves_schema)
    #featuresDF = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
    pass

if __name__ == "__main__":
    main()

