import torch 
import numpy as np 

boltzmann_constant = 1.380e-23
mu_0 = 1.256e-6 

def find_tesep(profs): 
    ne, te = profs[:, 0], profs[:, 1]
    teseps, neseps, rseps = np.empty(te.shape[0]), np.empty(te.shape[0]), np.empty(te.shape[0])
    for k, (ne_slice, te_slice) in enumerate(zip(ne, te)): 
        l_idx, r_idx = 0, 1
        while te_slice[r_idx] > 100: 
            l_idx += 1
            r_idx += 1 
            if r_idx == 50:#  or (len(te_slice) == 60 and r_idx == 60): 
                break 

        if r_idx == 50:
            continue 
        weights_r, weights_l = get_weights(te_slice, l_idx, r_idx)
        tesep_estimation = weights_l*te_slice[l_idx] + weights_r*te_slice[r_idx]
        nesep_estimation = weights_l*ne_slice[l_idx] + weights_r*ne_slice[r_idx]
        # rsep_estimation = weights_l*r_slice[l_idx] + weights_r*r_slice[r_idx]
        teseps[k] = tesep_estimation
        neseps[k] = nesep_estimation
        rseps[k] = l_idx
    # rsep_estimation = weights_l*x[idx_l] + weights_r*x[idx_r]
    return teseps, neseps, rseps

def get_weights(te, idx_l, idx_r, query=100):
    # Gets weighst as usual
    dist = te[idx_r] - query + query - te[idx_l]
    weights = (1-(te[idx_r] - query)/dist, 1-(query - te[idx_l])/dist)
    return weights

def static_pressure_stored_energy_approximation(profs):
    if not isinstance(profs, torch.Tensor): 
        profs = torch.from_numpy(profs)
    return boltzmann_constant*torch.prod(profs, 1).sum(1)
def torch_shaping_approx(minor_radius, tri_u, tri_l, elongation):
    triangularity = (tri_u + tri_l) / 2.0
    b = elongation*minor_radius
    gamma_top = -(minor_radius + triangularity)
    gamma_bot = minor_radius - triangularity
    alpha_top = -gamma_top / (b*b)
    alpha_bot = -gamma_bot / (b*b)
    top_int = (torch.arcsinh(2*torch.abs(alpha_top)*b) + 2*torch.abs(alpha_top)*b*torch.sqrt(4*alpha_top*alpha_top*b*b+1)) / (2*torch.abs(alpha_top))
    bot_int = (torch.arcsinh(2*torch.abs(alpha_bot)*b) + 2*torch.abs(alpha_bot)*b*torch.sqrt(4*alpha_bot*alpha_bot*b*b+1)) / (2*torch.abs(alpha_bot))
    return bot_int + top_int 

def bpol_approx(minor_radius, tri_u, tri_l, elongation, current): 
    shaping = torch_shaping_approx(minor_radius, tri_u, tri_l, elongation)
    return mu_0*current / shaping

def calculate_peped(profs): 
    if not isinstance(profs, torch.Tensor): 
        profs = torch.from_numpy(profs)
    ne = profs[:, 0:1, :]
    te = profs[:, 1:, :]
    p = boltzmann_constant*ne*te
    second_diff = torch.diff(p, n=2, dim=-1)
    min_diff_val, min_diff_idx = torch.min(second_diff, dim=-1)
    ped_loc = min_diff_idx -2
    peped = torch.zeros((len(p)))
    for k in range(len(peped)): 
        peped[k] = torch.select(p[k], dim=-1, index=ped_loc[k].item()) 
    return peped, ped_loc
def beta_approximation(profiles_tensors, minor_radius, tri_u, tri_l, elongation, current, bt, beta_pol=False):
    """
    To approximate beta! 
    The factor of 2 at the front is to compensate the ions which are nowhere to be found in this analysis. 
    The additional factor of 100 is to get it in percent form. 
    """
    e_c = 1.602e-19
    bpol = bpol_approx(minor_radius, tri_u, tri_l, elongation, current)
    if beta_pol: 
        pressure_ped, _ = calculate_peped(profiles_tensors)
        beta_pol_approx =  2*mu_0*pressure_ped / (bpol*bpol)
        return beta_pol_approx 
    density, temperature = profiles_tensors[:, 0, :], profiles_tensors[:, 1, :]
    pressure_prof = density*temperature
    pressure_average = pressure_prof.mean(-1)
    # TODO: This beta average is not really realistic I find... but am interested to see how it impacts
    return (100*2)*e_c*2*mu_0 * pressure_average / (bt*bt + bpol*bpol)

def pressure_calculation(profs: torch.Tensor, dataset=None, normalize=True): 
    if normalize and dataset is not None: 
        profs = dataset.denorm_profiles(profs.copy())
    return boltzmann_constant*torch.prod(profs, 1)

def calculate_physics_constraints(profiles_og, mps_og, train_set):
    # Denormalize everything! 
    profiles = torch.clone(profiles_og)
    profiles = train_set.denorm_profiles(profiles, to_torch=True)
    mps = torch.clone(mps_og)
    mps = train_set.denorm_mps(mps, to_torch=True)
    sp =  static_pressure_stored_energy_approximation(profiles)
    minor_radius, tri_u, tri_l, elongation, current, bt = mps[:, 2], mps[:, 4],mps[:, 5],mps[:, 6], mps[:, 8], mps[:, 9]
    bpol = bpol_approx(minor_radius, tri_u, tri_l, elongation, current)
    beta = beta_approximation(profiles, minor_radius, tri_u, tri_l, elongation, current, bt)
    pressure = pressure_calculation(profiles, normalize=False)

    return sp, beta, bpol, pressure
