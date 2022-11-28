# Data Overview


1. Collect the immutable raw data from the relevant labs (e.g., JET or AUG). 
2. Convert the raw data to full length numpy arrays 
3. Feed to model 


### 1. Collect raw data 

- **JET**: `Gather_From_JET.py`: To be run on Heimdall, to collect the raw immutable data straight from JET PPF's. 

This will generate a large number of files, each a python dictionary containing the shotfile with the following structure: 

`pulse_dict = {'profiles': profile_dict,  machine_parameters': mp_dict, 'journal': journal_dict}`

- `profile_dict {'time': time_base, 'radius': rhop, 'Te': Te, 'ne': Ne, 'Te_unc': Te_unc, 'ne_unc': Ne_unc}`
- `mp_dict[param] = {'time': time, 'data': data}` with `params` found in `['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'P_OH', 'PICR_TOT', 'PECR_TOT', 'P_TOT', 'SHINE_TH', 'Wmhd', 'Wfi', 'Wth', 'dWmhd/dt', tau_tot', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']`

These files should be stored under some directory, e.g., `/data/raw/`

Instantiate an `$RAW_DIR` environment variable via `$ export RAW_DIR=/home/user/moxie/data/raw/`, as well as a `$PROC_DIR` environment variable, such as `/data/processed`. These environment variables will be used in the rest of the analysis. 

### 2. Convert to numpy arrays 

Using the raw data, we map machine parameters to profiles by interpolating the machine parameters time wise to match profile diagnostic time stamps. Other than that, we grab the all available time slices. 
- **JET**: `build_arrays_JET.py`

See `python3 build_arrays_JET.py --help` for a list of arguments to pass, namely the location to store the . 

This will generate 3 arrays per pulse in a user specified folder under `$PROC_DIR`, with names e.g., `32339_t1=0.07_t2=6.71_PROFS'` for the profiles between 0.07 and 6.71s of pulse number 32339. The machine parameters and radii for the profiles will be stored under `_MP` and `_RADII` respectively. 

The labels of the columns in the machine parameter array are saved (in a `.txt` file) within the same created folder. 


### 3. Feed to model 

This will be done on the fly for model training. 

Here one should establish which set of machine parameters to use.  

- **'Observational'** model, which takes time-independent observations of the plasma state. 

Therefore, we have different datasets/classes for each. 