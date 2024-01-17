#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:49:32 2023

@author: ehua
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:25:19 2022

@author: huange
"""
from spynal.matIO import loadmat
import numpy as np
import pandas as pd
from analysis import *
from preprocessing import preProcessing, featExtract, coupling
import os

def select_area(unit_info, data, area_name):
    '''
    Isolates data recorded from a select area.
        
        Input: unit_info: dataframe containing unit recording information, such as area recorded
               spike_times (n_units, n_timepts): ndarray of spike timestamps
               spike_waves (n_units, n_timepts): ndarray of spike waveforms
               area_name: area of interest

        Output: spike times, waves of select area

    '''
    areas = unit_info['area'].to_numpy()
    area_idx = np.where(areas == area_name)[0]
    
    data = data[:, area_idx]

    return data, area_idx

def shape_data(spike_times, spike_waves = None):
    
    for i in range(spike_times.shape[0]):
       for j in range(spike_times.shape[1]):
           spike = spike_times[i,j]
           if spike_waves is not None:
                wave = spike_waves[i,j]
                if len(wave.shape) < 2:
                    wave = np.expand_dims(wave, axis = 1)
           if type(spike) != float:
               if np.size(spike) != 0:
                   trunc_spike = np.where((-1 < spike) & (spike < 2))[0]
                   
                   spike_times[i,j] = np.atleast_1d(spike[trunc_spike])

                   if spike_waves is not None:
                       trunc_wave = wave[:, trunc_spike]
                       spike_waves[i,j] = np.atleast_2d(trunc_wave)
               else:
                   spike_times[i, j] = []

                   if spike_waves is not None:
                       spike_waves[i,j] = np.empty((48,0))
           else:
               if -1 < spike < 2:
                   spike_times[i,j] = [spike]  

                   if spike_waves is not None:
                       spike_waves[i,j] = np.expand_dims(wave, 1)
               else:
                   spike_times[i,j] = []

                   if spike_waves is not None:
                       spike_waves[i,j] = np.empty((48, 0))
    
    return spike_times, spike_waves

def load_data(path):
    spike_times, spike_times_schema, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = \
    loadmat(path,
        variables=['spikeTimes','spikeTimesSchema','unitInfo','trialInfo', 'sessionInfo', 'spikeWaves', 'spikeWavesSchema'],
        typemap={'unitInfo':'DataFrame', 'trialInfo':'DataFrame'})\
    
    shape_spikes, shape_waves = shape_data(spike_times, spike_waves = spike_waves)
    
    return shape_spikes, spike_times_schema, unit_info, trial_info, session_info, shape_waves, spike_waves_schema

def load_osc_data(path):
    lfp, lfp_schema, electrode_info, spike_times, unit_info = \
    loadmat(path,
        variables=['lfp', 'lfpSchema', 'electrodeInfo', 'spikeTimes', 'unitInfo'],
        typemap={'electrodeInfo':'DataFrame', 'unitInfo':'DataFrame'})\
        
    spike_times, _ = shape_data(spike_times)
        
    return lfp, lfp_schema, electrode_info, spike_times, unit_info

def concat_sessions(paths, area):
    comb = pd.DataFrame(columns=['meanRates', 'troughToPeak', 'repolTime', 'CV', 'LV'])
    PEV_samp_concat = np.empty((0, 30))
    PEV_pred_concat = np.empty((0, 30))
    align_waves_concat = np.empty((470, 0))
    depths_concat = np.empty((0, ))
    pred_concat = np.empty((0, ))
    samp_concat = np.empty((0, ))
    unit_count = 0

    PEV_unpredSamp_concat = np.empty((0, 30))
    
    for path in paths:
        spike_times, _, unit_info, trial_info, session_info, spike_waves, spike_waves_schema = load_data(path)
        area_spike_times, area_idx = select_area(unit_info, spike_times, area)
        area_spike_waves = spike_waves[:, area_idx]
        
        validTrials, validNeurons, meanRates, ISIs, meanAlignWaves, smpRate, rates, _, _, _, predInfo, sampInfo, depths, _, _ = \
            preProcessing(area_spike_times, 
                          trial_info, 
                          session_info, 
                          area_spike_waves, 
                          spike_waves_schema, 
                          unit_info, 
                          area, 
                          unit_count) \
                          
        if validTrials.size < 2 or len(validNeurons) < 2:
            pass
        else:
            unit_count += len(validNeurons)
            features = featExtract(meanRates, ISIs, meanAlignWaves, smpRate, rates)
            comb = pd.concat([comb, features], ignore_index=True)
            #pev_samp = pev_func(rates, sampInfo)
            pev_pred = pev_func(rates, predInfo)
           
            #PEV_samp_concat = np.concatenate((PEV_samp_concat, np.squeeze(pev_samp, axis=0)), axis = 0)
            PEV_pred_concat = np.concatenate((PEV_pred_concat, np.squeeze(pev_pred, axis=0)), axis = 0)

            # pev_unpredSamp = pev_func(rates[trial_trials, :, :], unpredSampInfo)
            # PEV_unpredSamp_concat = np.concatenate((PEV_unpredSamp_concat, np.squeeze(pev_unpredSamp, axis=0)), axis = 0)

            align_waves_concat = np.concatenate((align_waves_concat, meanAlignWaves), axis = 1)

            depths_concat = np.concatenate((depths_concat, depths), axis = 0)

            pred_concat = np.concatenate((pred_concat, predInfo.to_numpy()), axis = 0)
            samp_concat = np.concatenate((samp_concat, sampInfo.to_numpy()), axis = 0)

    return comb, PEV_pred_concat, align_waves_concat, depths_concat, pred_concat, samp_concat


def osc_concat(paths, areas):
    session = 0
    for path in paths:
        lfp, lfp_schema, electrode_info, spike_times, unit_info = load_osc_data(path)
        
        for area in areas:
            if unit_info['area'].isin([area]).any():
                area_lfp, area_idx = select_area(electrode_info, lfp, area)
                lfp_trunc = area_lfp[1000:4001, :, :]
                area_spikes = spike_times[:, area_idx]

                smp_rate = lfp_schema['smpRate']

                if '/mnt/common/datasets/wmPredict/mat/mainTask' in path:
                    depth_var = 'betaGammaDepth'
                else:
                    depth_var = 'laminarDepth'

                coupling(lfp_trunc, area_idx, depth_var, electrode_info, unit_info, area_spikes, area, session, smp_rate)

        session += 1
        
def main(): 
    directories = ['/mnt/common/datasets/wmPredict/mat/mainTask', '/mnt/common/scott/laminarPharm/mat']
    paths = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename == 'laminarPharm_databases.mat' or filename == 'spikesOnly' or filename == 'wmPredict_databases.mat':
                pass
            else:
                f = os.path.join(directory, filename)
                paths.append(f)
    
    areas = ['vlPFC', 'dlPFC', '7A', 'V4', 'LIP']
        
        #comb, PEV_pred, waves, depths, pred, samp = concat_sessions(paths, area)

    osc_concat(paths, areas)

        # unpredSamp_df = pd.DataFrame(PEV_samp)
        # unpredSamp_df.to_csv('/home/ehua/clustering/090623_data/{}_PEV_unpredSamp.csv'.format(area))
        
        # df = pd.DataFrame(comb)
        # df.to_csv('/home/ehua/clustering/090623_data/{}_df.csv'.format(area))
        
        # waves_df = pd.DataFrame(waves)
        # waves_df.to_csv('/home/ehua/clustering/090623_data/{}_waves.csv'.format(area))
        
        # samp_df = pd.DataFrame(PEV_samp)
        # samp_df.to_csv('/home/ehua/clustering/090623_data/{}_PEV_samp.csv'.format(area))
        
        # pred_df = pd.DataFrame(PEV_pred)
        # pred_df.to_csv('/home/ehua/clustering/090623_data/{}_PEV_pred.csv'.format(area))

        # depths_df = pd.DataFrame(depths)
        # depths_df.to_csv('/home/ehua/clustering/090623_data/{}_depths_jitter.csv'.format(area))

        # pred_df = pd.DataFrame(pred)
        # pred_df.to_csv('/home/ehua/clustering/090623_data/{}_pred.csv'.format(area))

        # samp_df = pd.DataFrame(samp)
        # samp_df.to_csv('/home/ehua/clustering/090623_data/{}_samp.csv'.format(area))


if __name__ == "__main__":
    main()
    
    