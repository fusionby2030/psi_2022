"""
author: adam.kit@helsinki.fi

Script uses the Simple Access Layer from JET to gather the required data.
"""
from jet.data import sal
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
CURRENT_DATE = datetime.today().strftime('%d%m%Y')

def get_ppf(pulse, dda, dtype, user='jetppf', sequence=0):
    return sal.get('/pulse/{}/ppf/signal/{}/{}/{}:{}'.format(pulse, user, dda, dtype, sequence))

def get_control(signal):
    data = {}
    for dim in signal.dimensions:
        if dim.temporal:
            data['time'] = dim.data
        else:
            data['radius'] = dim.data
    data['values'] = signal.data
    return data

def get_profile(signal):
    data = {}
    for dim in signal.dimensions:
        if dim.temporal:
            data = dim.data
        else:
            data['radius'] = dim.data
    data['values'] = signal.data
    return data

def gather_all_inputs(pulse_num):
    name_dict = {'EFIT': ['Q95', 'RGEO', 'CR0', 'VOLM', 'TRIU', 'TRIL', 'ELON', 'POHM', 'VJAC', 'PSNM'],
                 'MAGN': ['IPLA', 'BVAC'],
                 'GASH': ['ELER'],
                 'NBI': ['NBLM'],
                 'ICRH': ['PTOT'],
                 'EDG8': ['TBEO']}
    if pulse_num < 80000:
        name_dict.pop('GASH')
        name_dict['GAS'] = ['ELER']
        if pulse_num < 80128:
            name_dict.pop('EDG8')

    control_dict = {}
    for dda, names in name_dict.items():
        try:
            for name in names:
                control_signal = get_ppf(pulse_num, dda, name)
                input_dict = get_control(control_signal)
                if dda == 'NBI' or dda == 'ICRH':
                    control_dict[dda] = input_dict
                else:
                    control_dict[name] = input_dict
        except Exception:
            if dda == 'EFIT':
                for name in names:
                    control_signal = get_ppf(pulse_num, 'EHTR', name)

                    input_dict = get_control(control_signal)

                    control_dict[name] = input_dict
            if dda == 'ICRH':
                control_dict['ICRH'] = {'time': 'NO_ICRH_USED', 'values': np.zeros(100)}
            if dda == 'EDG8':
                control_dict['TBEO'] = {'time': 'BERLYM', 'values': np.zeros(100)}
            continue
    return control_dict

def gather_all_outputs(pulse_num):
    name_list = ['NE', 'DNE', 'DTE','TE', 'RHO', 'RMID', 'PSI', 'FAIL']
    name_dict = {'NE': 'ne', 'DNE': 'ne_unc', 'TE': 'Te', 'DTE': 'Te_unc',
                'PSI': 'radius', 'RHO': 'rhop'}
    profiles_dict = {}
    for name in name_list:
        try:
            profile_signal = get_ppf(pulse_num, 'HRTS', name)
            single_dict = get_profile(profile_signal)
            profiles_dict[name_dict[name]] = single_dict
        except Exception:
            profiles_dict[name_dict[name]] = 'Unavailable'
            continue
    return profiles_dict

def gather_elm_timings(pulse_num, dda, uid):
    try:
        new_signal = get_ppf(pulse_num, dda=dda, user=uid, dtype='TEL1')
        new_data = new_signal.data
        return new_data
    except Exception as e:
        print('NO ELM TIMINGS')
        return np.array(['NO ELM TIMINGS'])

if __name__ == '__main__':
    df = pd.read_csv('/home/mn2596/JETPEDESTAL_ANALYSIS/moxie_profile_project/final_data/processed/jet-pedestal-database.csv')
    validated_df = df[df['FLAG:HRTSdatavalidated'] > 0]

    valid_count = 0
    all_dict = {}

    for index, row in validated_df.iterrows():
        name, pulse_num = row['dda'], int(row['shot'])
        dda, uid = name.split('/')
        valid_count += 1
        if str(pulse_num) not in all_dict.keys():
            print('\n-----SHOT {} -------{} Total'.format(pulse_num, valid_count))
            print('Gathering Outputs')
            outputs_dict = gather_all_outputs(pulse_num)
            print('Gathering Inputs')
            inputs_dict = gather_all_inputs(pulse_num)
            print('Gathering ELMS')
            elm_arr = gather_elm_timings(pulse_num, dda, uid)
            all_dict[str(pulse_num)] = {'machine_parameters': inputs_dict, 'profiles': outputs_dict, 'elms': elm_arr}
        else:
            elm_arr = gather_elm_timings(pulse_num, dda, uid)
            if elm_arr != 'NO ELM TIMINGS':
                all_dict[str(pulse_num)]['elms'] = np.append(all_dict[str(pulse_num)]['elms'], elm_arr)
    with open(f'../../data/raw/JET_RAW_DATA_{CURRENT_DATE}.pickle', 'wb') as file: # f'processed_pulse_dict_{CURRENT_DATE}.pickle'
            pickle.dump(all_dict, file)
