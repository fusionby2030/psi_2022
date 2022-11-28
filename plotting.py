"""
Use with saved model. 
Produces the current sweep and latent space plots
"""
from src.models import DIVA
from src.data import MemMapDataset_O
from src.data._utils import get_dataloaders 
from src.common.utils import load_model 
from src.common.physics_approximations import *
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl

MP_names_JET = ['BTF', 'D_tot', 'IpiFP', 'PNBI_TOT', 'P_OH', 'PICR_TOT', 'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol', 'elm_timings']
def main(model_name): 
    state_dict, hparams, dataset = load_model(model_name)
    model = DIVA(**hparams)
    model.load_state_dict(state_dict) 
    model.double()
    current_sweep_plot(model, dataset)
    latent_space_plot(model, dataset)

def latent_space_plot(model, dataset): 

    Z_MACH, Z_STOCH = [], []
    for idx in range(dataset.total_num_pulses // 2): 
        sample_profs, sample_mps = torch.from_numpy(dataset.data['profs'][idx].copy()), torch.from_numpy(dataset.data['mps'][idx].copy())
        sample_profs_norm, sample_mps_norm = dataset.norm_profiles(sample_profs), dataset.norm_mps(sample_mps)

        with torch.no_grad(): 
            _,z_mach, z_stoch, *_ = model.prof2z(sample_profs_norm)
            Z_MACH.extend(z_mach)
            Z_STOCH.extend(z_stoch)

    Z_MACH, Z_STOCH = torch.vstack(Z_MACH), torch.vstack(Z_STOCH)
    image_res = 512
    sample_size = image_res ** 2 # 2D
    r1, r2 = -5, 5
    a, b = sample_size, 2
    ld_1, ld_2 = 0, 2

    range_xy = torch.linspace(start=r1, end=r2, steps=image_res)
    range_xy = torch.cartesian_prod(range_xy, range_xy)
    range_imagecoord = torch.linspace(0, image_res-1, steps=image_res, dtype=torch.int32)  # so we can easily go back
    range_imagecoord = torch.cartesian_prod(range_imagecoord, range_imagecoord)

    
    z_mach_mean, z_stoch_mean = Z_MACH.mean(0), Z_STOCH.mean(0)
    z_mach_sample, z_stoch_sample = torch.tile(z_mach_mean, (sample_size, 1)), torch.tile(z_stoch_mean, (sample_size, 1))

    z_mach_sample[:, ld_1] = range_xy[:, 0]
    z_mach_sample[:, ld_2] = range_xy[:, 1]
    image_array = np.zeros((image_res, image_res))

    with torch.no_grad():
        z_conditional = torch.cat((z_mach_sample, z_stoch_sample), 1)
        out_profs = model.z2prof(z_conditional)
        out_mps = model.z2mp(z_mach_sample)

    sample_profs, sample_mps = dataset.denorm_profiles(out_profs), dataset.denorm_mps(out_mps)
    sample_teseps, sample_neseps, sample_rseps = find_tesep(sample_profs) 

    image_array = np.zeros((image_res, image_res))

    for i in range(range_imagecoord.shape[0]):
        _x, _y = range_imagecoord[i]
        _y = image_res - 1 - _y  # (0, 0) for img are on top left so reverse
        image_array[_y, _x] = sample_neseps[i]

    sample_zx, sample_zy = 2, 0 #The star on th graph 

    data_sample = torch.tensor([sample_zx, sample_zy])
    min_i = -1
    min_dist = 100000
    for i in range(range_xy.shape[0]):
        a = data_sample.cpu().numpy()
        b = range_xy[i].cpu().numpy()
        dist = np.linalg.norm(a-b)
        if dist < min_dist:
            min_dist = dist
            min_i = i

    sample_1 = min_i

    fig, ls_ax,= plt.subplots(constrained_layout=True)

    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=0, vmax=1e20)
    """ LATENT SPACE PLOT """
    cax = ls_ax.imshow(np.rot90(image_array, 3), extent=[r1, r2, r1, r2], cmap=cmap, norm=norm, interpolation='spline36')
    fig.colorbar(cax, ax=ls_ax, label='Inferred $n_e^{sep}$ [m$^{-3}$]', location='left')
    ls_ax.set_xlabel('Latent Dimension 4')
    ls_ax.set_ylabel('Latent Dimension 6')
    plt.show()

def current_sweep_plot(model, dataset):
    MEAN_MP = [] 
    for idx in range(dataset.total_num_pulses // 2): 
        sample_profs, sample_mps = torch.from_numpy(dataset.data['profs'][idx].copy()), torch.from_numpy(dataset.data['mps'][idx].copy())
        MEAN_MP.append(sample_mps)
    MEAN_MP = torch.vstack(MEAN_MP).mean(0)
    N_SAMPLES = 1000
    current_sweep = torch.linspace(1e6, 5e6, N_SAMPLES)

    MP_IN = torch.tile(MEAN_MP, (N_SAMPLES, 1))
    MP_IN[:, 2] = current_sweep

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=6e6)

    with torch.no_grad(): 
        out_profs_norm, _, _ = model.inference(dataset.norm_mps(MP_IN), from_mean=False)
        out_profs = dataset.denorm_profiles(out_profs_norm)
    fig = plt.figure()
    
    for k, sample in enumerate(out_profs): 
        plt.plot(sample[0], color=cmap(norm(current_sweep[k])))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),orientation='horizontal', label='$I_P$ [MA]')
    plt.show()
    


if __name__ == '__main__': 
    model_name = 'DIVA.pth'
    main(model_name)