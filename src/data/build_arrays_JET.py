import pickle, os, argparse
from typing import Any, Dict, List, Tuple, Union 
from tqdm import tqdm 
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
import utils_ as utils
import aug_sfutils
import psutil
from multiprocessing import Pool


REL_JET_COLS = ['BTF', 'D_tot', 'IpiFP', 'PNBI_TOT', 'P_OH', 'PICR_TOT', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol', 'elm_timings']

def build(file_list: List[str]) -> None: 
    """ Build the relevant numpy files! 

    The numpy arrays will be stored with the shot number (filename in this case), followed by a _PROFS, _MPS, _RADII depending if it is the profiles, machine parameters, or radii
    Parameters
    ----------
    file_list : List[str]
        List of raw AUG data files to pull from. 
        These are generated from AUG_SFUTILS, which will be linked somewhere!  
    """
    global jet_pdb
    jet_pdb = pd.read_csv(RAW_DIR + 'jet-pedestal-database.csv')
    if args.mp: 
        print(f'Doing multiprocessing with {psutil.cpu_count(logical=False)} cpus')
        pool = Pool(psutil.cpu_count(logical=False))
        pool.map(convert_raw_file_to_array, file_list) 
    else: 
        for filename in tqdm(file_list, total=len(file_list)): 
            convert_raw_file_to_array(filename)

    # To save the names of the machine parameters / column order somewhere
    with open(PROC_DIR + f'/{args.array_folder_name}/mp_column_names.txt', 'w') as file: 
        for col in REL_JET_COLS: 
            file.write(f'{col}, ') 

def convert_raw_file_to_array(filename): 
    global t1, t2
    shot_num: str = filename.split('/')[-1]
    with open(filename, 'rb') as file: 
        pulse_dict: Dict[str, Dict[str, np.ndarray]] = pickle.load(file)

    rel_jpdb_rows = jet_pdb[jet_pdb['shot'] == int(shot_num)]
    for idx, row in rel_jpdb_rows.iterrows(): 
        t1, t2 = float(row['t1']), float(row['t2'])
        profile_data, mp_data, journal_data = pulse_dict['profiles'], pulse_dict['machine_parameters'], None
        
        if profile_data['ne'] is None or profile_data['Te'] is None or profile_data['radius'] is None: 
            return None
        relevant_profiles, relevant_mps, relevant_radii, [_, _], relevant_times = gather_relevant_arrays(profile_data, mp_data)
        if relevant_profiles is None or relevant_mps is None or relevant_radii is None or t1 is None:
            return None
        save_shot_arrays(relevant_profiles, relevant_mps, relevant_radii, relevant_times, shot_num, exp_name=f't1={t1:.2f}_t2={t2:.2f}', final_journal=journal_data)


def map_mps_to_profs_JET(ida_times: np.ndarray, mp_data: dict, window_size: float = 0.002, relevant_mp_columns: List[str] = REL_JET_COLS) -> np.ndarray:
    """Converts dictionary of raw machine parameters to a suitable numpy array by mapping machine parameters at the relevant ida time slices
    
    Mapping: 
    For each IDA time stamp, for each machine parameter
         a) find a corresponding mp time window given the IDA_time stamp, simply this is between [time_stamp - window_size, time_stamp + window_size]
         b) average the values in the window 
         c) if there are no mps in the given window, expand the window by 2 ms. 
    Parameters
    ----------
    ida_times : np.ndarray
        The relevant IDA time stamps
    mp_data : dict
        dictionary of machine parameter data 
    window_size : float, optional
        window in which to average machine parameters, by default 0.002 (ms)
    relevant_mp_columns : List[str], optional 
        The relevant MP cols to store from the raw data 

    Returns
    -------
    relevant_machine_parameters np.ndarray
        this will have length of the relevant profiles! 
    """
    def map_mp_to_time_stamp(time_stamp) -> Any:
        """Take an ida/hrts time stamp, and use the interpolated machine parameters to collected a window of the MP, and mean that window 
        """
        # xtim: np.ndarray = np.linspace(max(time_stamp-window_size, min(ida_times)), min(time_stamp+window_size, max(ida_times)),10)
        xtim: np.ndarray = np.linspace(time_stamp - window_size, time_stamp+window_size,10)
        window: np.ndarray = f(xtim)
        val = np.mean(window)
        return val 
    def get_elm_timings(pulse_elms, time_windows):
        def calculate_elm_percent(time_last_elm, time_next_elm, hrts_time): 
            return (hrts_time - time_last_elm) / (time_next_elm - time_last_elm)
        elm_timings = np.empty(len(time_windows))
        elm_timings[:] = np.nan
        for t_idx, hrts_time in enumerate(time_windows):
            diff = pulse_elms - hrts_time
            try: 
                time_last_elm = pulse_elms[diff < 0][-1]
                time_next_elm = pulse_elms[diff > 0][0]
            except IndexError as e: 
                continue 
            else: 
                slice_elm_percent = calculate_elm_percent(time_last_elm, time_next_elm, hrts_time)
                elm_timings[t_idx] = slice_elm_percent
        return elm_timings

    map_ts = np.vectorize(map_mp_to_time_stamp)
    relevant_machine_parameters: np.ndarray = np.empty((len(ida_times), len(relevant_mp_columns)))
    relevant_machine_parameters[:] = np.nan
    
    for mp_idx, key in enumerate(relevant_mp_columns): 
        if key == 'elm_timings': 
            elm_data = np.array(mp_data[key])
            if len(elm_data) == 1 and elm_data[0] == 0.0: 
                raise ValueError('no elm timings')            
            else: 
                relevant_machine_parameters[:, mp_idx] =  get_elm_timings(elm_data, ida_times)
                continue
        relevant_mp_vals = np.zeros(len(ida_times))
        try: 
            mp_raw_data, mp_raw_time = mp_data[key]['data'], mp_data[key]['time']
        except KeyError as e: 
            relevant_machine_parameters[:, mp_idx] = relevant_mp_vals

        if mp_raw_time is None: 
            relevant_machine_parameters[:, mp_idx] = relevant_mp_vals
            continue 
        elif len(mp_raw_time) != len(mp_raw_data) and key == 'PICR_TOT': 
            relevant_machine_parameters[:, mp_idx] = relevant_mp_vals
            continue 
        else:
            f = interp1d(mp_raw_time, mp_raw_data)
            relevant_mp_vals = map_ts(ida_times)

        relevant_machine_parameters[:, mp_idx] = relevant_mp_vals
    return relevant_machine_parameters 

def remap_profiles_jet(relevant_profiles: np.ndarray, relevant_radii: np.ndarray): 
    num_map_radius = 50

    # mapping_radius = np.linspace(0.8, 1.1, num_map_radius)
    remapped_profiles, remapped_radii = np.empty((relevant_profiles.shape[0], relevant_profiles.shape[1], num_map_radius)), np.empty((relevant_profiles.shape[0], num_map_radius))
    for k, (slice_prof, slice_rad) in enumerate(zip(relevant_profiles, relevant_radii)): 
        ne_slice, te_slice = np.copy(slice_prof[0]), np.copy(slice_prof[1])  
          
        mapping_radius = np.linspace(0.8, max(slice_rad), num_map_radius)
        ind = np.where(ne_slice<5e18)
        ind1 = np.where(te_slice > 50)
        index = np.intersect1d(ind,ind1)
        te_slice[index] = ne_slice[index]*50/5e18

        f_te = interp1d(slice_rad, te_slice)
        f_ne = interp1d(slice_rad, ne_slice)
        try: 
            psi_tesep_idx =  (np.abs(f_te(mapping_radius) - 100)).argmin()            
        except ValueError as e: 
            import matplotlib.pyplot as plt 
            plt.plot(slice_rad, ne_slice)
            plt.show()
            plt.plot(slice_rad, te_slice)
            plt.show()
            print(slice_rad)
            print('wtf')
            raise e
        psi_tesep = mapping_radius[psi_tesep_idx]
        psi_shift = 1.0 - psi_tesep

        remaped_radius = slice_rad + psi_shift

        f_te = interp1d(remaped_radius, te_slice)
        f_ne = interp1d(remaped_radius, ne_slice)
        new_mapping_radius = np.linspace(0.8, max(remaped_radius), num_map_radius)
        fitted_te = f_te(new_mapping_radius)
        fitted_ne = f_ne(new_mapping_radius)
        
        remapped_profiles[k, 0, :] = fitted_ne
        remapped_profiles[k, 1, :] = fitted_te 

        remapped_radii[k] = new_mapping_radius       
    return remapped_profiles, remapped_radii

def gather_relevant_arrays(profile_data: dict, mp_data: dict) -> Tuple[np.ndarray]: 
    """ Gather the relevant profile and machine parameter arrays from IDA times 

    Parameters
    ----------
    profile_data : dict
        coming from the raw AUG_SFUTILS file, this has keys
        'ne', 'Te', 'ne_unc', 'Te_unc', 'time', 'radius'
    mp_data : dict
        coming from the raw AUG_SFUTILS file, this has keys
        ALOT OF STRINGS

    Returns
    -------
    relevant profiles, relevant_machine_parameters relevant_radii, : Tuple[np.ndarray]
        profiles: shape [#Slices, 2, 75]
        machine_parameters: shape [#Slices, 14] (well 14 depends on what you define the )
    """

    ida_times, ne, te, radius = np.array(profile_data['time']), np.array(profile_data['ne']), np.array(profile_data['Te']), np.array(profile_data['radius'])
    profiles = np.stack((ne, te), 1)

    relevant_time_windows_bool = np.logical_and(ida_times > t1, ida_times<t2)
    # relevant_time_windows_bool = np.array([True]*len(ida_times))
    relevant_time_windows: np.ndarray = ida_times[relevant_time_windows_bool]
    # t1, t2 = ida_times[0], ida_times[-1]
    relevant_profiles = profiles[relevant_time_windows_bool]
    relevant_radii = radius[relevant_time_windows_bool]
    relevant_profiles, relevant_radii = remap_profiles_jet(relevant_profiles, relevant_radii)
    try: 
        relevant_machine_parameters = map_mps_to_profs_JET(relevant_time_windows, mp_data)
        non_nan_time_windows = np.invert(np.isnan(relevant_machine_parameters[:, -1]))
        relevant_profiles, relevant_machine_parameters, relevant_radii, relevant_time_windows = relevant_profiles[non_nan_time_windows], relevant_machine_parameters[non_nan_time_windows], relevant_radii[non_nan_time_windows], relevant_time_windows[non_nan_time_windows]
        if len(relevant_profiles) == 0: 
            return None, None, None, [None, None], None
    except IndexError as e: 
        print(e)
        print('fuck')
        return None, None, None, [None, None], None
    except ValueError as e: 
        print(e)
        return None, None, None, [None, None], None
    else: 
        return relevant_profiles, relevant_machine_parameters, relevant_radii, [t1, t2], relevant_time_windows
 

def save_shot_arrays(final_profiles:np.ndarray, final_mps: np.ndarray, final_radii: np.ndarray, final_times: np.ndarray, shot_num: int, exp_name: str, final_journal: dict=None) -> None: 
    """Save the shot arrays to a numpy file

    Also saves the jounral dict 
    as well as a file that has the relevant machine parameters
    Parameters
    ----------
    final_profiles : np.ndarray
        profiles that are return from gather_relevant_arrays
    final_mps : np.ndarray
        mps that are return from gather_relevant_arrays
    final_radii : np.ndarray
        radii that are return from gather_relevant_arrays
    final_journal : dict 
    shot_num : int
        Corresponding shot number
    exp_name : str
        actually the relevant times of the pulse

    """
    relevant_path = PROC_DIR + f'/{args.array_folder_name}/{shot_num}_' + exp_name
    
    with open(relevant_path + '_MP.npy', 'wb') as mp_file: 
        np.save(mp_file, final_mps)
    with open(relevant_path + '_PROFS.npy', 'wb') as prof_file: 
        np.save(prof_file, final_profiles)
    with open(relevant_path + '_RADII.npy', 'wb') as radii_file: 
        np.save(radii_file, final_radii)
    with open(relevant_path + '_TIME.npy', 'wb') as time_file: 
        np.save(time_file, final_times)
    
    if final_journal is not None: 
        journal_df = pd.DataFrame(final_journal, index=[0])
        journal_df.to_csv(relevant_path + '_JOURNAL.csv')

    print(f'Saved to {relevant_path}')
import argparse 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Process raw data into numpy arrays')
    parser.add_argument('-rf', '--raw_folder_name', type=str, default='RAW_AUG_PULSES', help='Folder name under which the raw immutable pulse data is stored. This will be found under your raw dir is.')
    parser.add_argument('-af', '--array_folder_name', type=str, default='ARRAY_STANDALONE', help='Folder name under which to store the raw numpy arrays. This will be found in whatever your processed dir is.')
    parser.add_argument('-mp', action='count', default=0, help='To do multiprocessing or not')
    args = parser.parse_args()
    
    RAW_DIR = os.getenv('RAW_DIR') or '/home/kitadam/ENR_Sven/moxie/data/raw/'
    PROC_DIR = os.getenv('PROC_DIR') or '/home/kitadam/ENR_Sven/moxie/data/processed/'

    file_dir = RAW_DIR + f'{args.raw_folder_name}/'
    list_pulse_files = sorted([os.path.join(file_dir, fname) for fname in os.listdir(file_dir) if not fname.endswith(".py")])

    print(f'\nRaw dir under which data will be pulled {RAW_DIR}\nProcessed dir under which things will be stored: {PROC_DIR}')

    if args.array_folder_name == 'ARRAY_STANDALONE': 
        print(f'No folder name specified to store/pull processed numpy data, using default under -> {PROC_DIR + args.array_folder_name} ')
    else: 
        print(f'numpy data will be stored/pulled in/from -> {PROC_DIR + args.array_folder_name} ')

    utils.check_dir_and_or_make_dir(PROC_DIR + f'/{args.array_folder_name}/', make_dir=True)
    
    
    build(file_list=list_pulse_files)