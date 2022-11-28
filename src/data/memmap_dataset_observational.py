from torch.utils.data import Dataset
from mmap_ninja.ragged import RaggedMmap
import os 
import numpy as np
from typing import Tuple
import torch 
    
def compute_raggedmmap(data_dir: str, data_string: str, batch_size: int = 128) -> RaggedMmap:
    """Compute the ragged map 

    Parameters
    ----------
    data_dir : str
        Base dir where arrays are stored and where mmaps will be stored
    data_string : str
        PROFS, MP, RADII, etc., ,
    batch_size : int, optional
        I think this is actually irrelevant... by default 128


    Returns
    -------
    RaggedMmap
        Shout out to mmap_ninja
    """
    def load_data(paths):
        for path in paths: 
            yield np.load(path)

    # TODO: Change the string to include interface Memmap
    save_name = os.path.join(data_dir, f'{data_string}_MMAP')
    if os.path.exists(save_name): 
        ragged_mmap = RaggedMmap(save_name)
        print(f'A ragged map for {data_string} exists at {save_name} with length: {len(ragged_mmap)}')
    else: 
        relevant_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(f"_{data_string}.npy")])
        ragged_mmap = RaggedMmap.from_generator(out_dir=save_name, 
                                             sample_generator=load_data(relevant_paths),
                                             batch_size=batch_size,
                                             verbose=True)
        print(f'Ragged mmap for {data_string} saved to {save_name}')
    return ragged_mmap

class MemMapDataset_O(Dataset): 
    """A mememory mapped dataset. 

    """
    def __init__(self, data_path: str, device: str = 'cpu', filter_mps = None): 
        self.data = {} 
        self.data['profs'] = compute_raggedmmap(data_path, data_string='PROFS')    
        self.data['mps'] = compute_raggedmmap(data_path, data_string='MP')    

        self.total_num_pulses = len(self.data['profs'])
        self.list_num_slices: list = [len(pulse) for pulse in self.data['profs']]
        self.cumsum_num_slices: np.ndarray = np.cumsum(self.list_num_slices) - 1
        self.total_num_slices = sum(self.list_num_slices)

        self.prof_length = self.data['profs'][0].shape[-1]
        self.mp_dim = self.data['mps'][0].shape[-1]
        self.factor = np.ones((2,self.prof_length))
        self.factor[0, :] *= 1e-19

        self.device = None

        self.filter_mps = filter_mps

    def get_pulse_idx(self, idx: int) -> Tuple[int, int]:
        """A fun function that calculates which slice to take from the mmap
        Since the mmap is (imo) a list of pulses, we need to find which pulse the queried idx is coming from. 
        This is calculated by looking at the,m  minimum value of the cumulative sum of all the slices across pulses that are >= idx 
        Then the internal pulse slice idx is just queried idx subtracted by the cumulative sum up to that pulse ( + 1)

        Parameters
        ----------
        idx : int
            a given slice, which can take on values of [0 -> total_num_slices]

        Returns
        -------
        Tuple[int, int] : (pulse_idx, slice_idx)
        """
        # list_num_slices: list = [len(pulse) for pulse in self.prof_mmap]
        # cumsum_num_slices: np.ndarray = np.cumsum(self.list_num_slices) - 1
        pulse_idx: int = np.where(self.cumsum_num_slices >= idx)[0][0]
        slice_idx: int = (idx - (self.cumsum_num_slices[pulse_idx] + 1))
        return pulse_idx, slice_idx

    def get_norms(self, relevant_idxs, return_norms=True) -> None: 
        profs = self.data['profs'][relevant_idxs]
        mps = self.data['mps'][relevant_idxs]
        self.prof_means = np.mean(np.stack([np.mean(prof*self.factor, axis=0) for prof in profs], 0), 0)
        self.prof_stds = np.mean(np.stack([np.std(prof*self.factor, axis=0) for prof in profs], 0), 0)
        self.mp_means = np.mean(np.stack([np.mean(mp, axis=0) for mp in mps], 0), 0)
        self.mp_stds = np.mean(np.stack([np.std(mp, axis=0) for mp in mps], 0), 0)
        if self.device is not None: 
            self.prof_means, self.prof_stds, self.mp_means, self.mp_stds = self.prof_means.to(self.device), self.prof_stds.to(self.device), self.mp_means.to(self.device), self.mp_stds.to(self.device)
        return self.return_norms()
    
    def set_norms(self, prof_means, prof_stds, mp_means, mp_stds) -> None: 
        self.prof_means, self.prof_stds, self.mp_means, self.mp_stds = prof_means, prof_stds, mp_means, mp_stds

    def return_norms(self) -> dict: 
        return dict(prof_means=self.prof_means, prof_stds=self.prof_stds, mp_means=self.mp_means, mp_stds=self.mp_stds)

    def norm_profiles(self, profiles):
        return (profiles*self.factor - self.prof_means) / self.prof_stds

    def denorm_profiles(self, profiles, to_torch=False): 
        if not to_torch: 
            return ((profiles*self.prof_stds) + self.prof_means) / self.factor
        else: 
            return ((profiles*torch.from_numpy(self.prof_stds)) + torch.from_numpy(self.prof_means)) / torch.from_numpy(self.factor)

    def norm_mps(self, mps): 
        return (mps - self.mp_means) / self.mp_stds
    
    def denorm_mps(self, mps, to_torch=False): 
        if to_torch: 
            return (mps*torch.from_numpy(self.mp_stds)) + torch.from_numpy(self.mp_means)
        if self.filter_mps is not None: 
            # mps = self.filter_mps(mps)
            return (mps*self.filter_mps(self.mp_stds)) + self.filter_mps(self.mp_means)

        else: 
            return (mps*self.mp_stds) + self.mp_means

    def __len__(self): 
        return self.total_num_slices
    
    def __getitem__(self, idx):
        pulse_idx_to_take_from, slice_idx_to_take_from = self.get_pulse_idx(idx) 
        sample_pulse_profs = self.data['profs'][pulse_idx_to_take_from]
        sample_pulse_mps = self.data['mps'][pulse_idx_to_take_from]
        profs = sample_pulse_profs[slice_idx_to_take_from]
        mps = sample_pulse_mps[slice_idx_to_take_from]
        profs, mps = self.norm_profiles(profs), self.norm_mps(mps)
        if self.filter_mps is not None: 
            mps = self.filter_mps(mps)
        return profs, mps
          