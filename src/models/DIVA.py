import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Dict, List, Tuple
from .model_utils import PRIORREG, AUXREG, ENCODER, DECODER
from src.common.physics_approximations import *

MIN_STDDEV = 1e-15

class DIVA(nn.Module): 
    def __init__(self,  
                mach_latent_dim: int, stoch_latent_dim: int,
                conv_filter_sizes: List[int],  mp_layer_sizes: List[int], 
                input_dim: int = 2, out_length: int = 50, action_dim: int = 15, 
                clamping_zero_tensor: torch.Tensor = None,
                BETA_KLD_COND: float=0.001,  BETA_KLD_STOCH: float=0.001, BETA_KLD_MACH: float=0.001, 
                GAMMA_PROF: float=100.0, GAMMA_MP: float=100.0, 
                LAMBDA_PRESSURE: float=10.0, LAMBDA_BETA: float=1.0, LAMDBA_BPOL: float = 1.0, 
                physics: bool = True, **kwargs):
        super(DIVA, self).__init__()

        self.physics = physics

        self.mach_latent_dim, self.stoch_latent_dim = mach_latent_dim, stoch_latent_dim
        self.conv_filter_sizes, self.mp_encoder_layer_sizes = conv_filter_sizes, mp_layer_sizes
        self.trans_conv_filter_sizes, self.mp_decoder_layer_sizes = conv_filter_sizes[::-1], mp_layer_sizes[::-1]
        
        self.prof_len, self.mp_dim = out_length, action_dim

        self.prof_encoder = ENCODER(filter_sizes=self.conv_filter_sizes, in_ch=input_dim, in_length=self.prof_len) 
        self.prof_decoder = DECODER(filter_sizes=self.trans_conv_filter_sizes, end_conv_size=self.prof_encoder.end_conv_size, clamping_zero_tensor=clamping_zero_tensor) 

        self.mp_encoder = PRIORREG(in_dim=self.mp_dim, out_dim=self.mach_latent_dim, hidden_dims=self.mp_encoder_layer_sizes, make_prior=True)
        self.mp_decoder = AUXREG(in_dim = self.mach_latent_dim, out_dim=self.mp_dim, hidden_dims=self.mp_decoder_layer_sizes)

        in_prior_size = self.prof_encoder.end_conv_size*self.conv_filter_sizes[-1]
        self.z_mu_mach = nn.Linear(in_prior_size, self.mach_latent_dim)
        self.z_var_mach = nn.Linear(in_prior_size, self.mach_latent_dim)

        self.z_mu_stoch = nn.Linear(in_prior_size, self.stoch_latent_dim)
        self.z_var_stoch = nn.Linear(in_prior_size, self.stoch_latent_dim)

        self.decoder_input = nn.Linear(self.stoch_latent_dim + self.mach_latent_dim, self.trans_conv_filter_sizes[0]*self.prof_encoder.end_conv_size)
        self.output_layer = nn.Linear(self.prof_decoder.final_size, out_length)

        self.GAMMA_PROF, self.GAMMA_MP = GAMMA_PROF, GAMMA_MP
        self.BETA_KLD_COND, self.BETA_KLD_STOCH , self.BETA_KLD_MACH =  BETA_KLD_COND, BETA_KLD_STOCH, BETA_KLD_MACH
        self.LAMBDA_PRESSURE, self.LAMBDA_BETA, self.LAMDBA_BPOL = LAMBDA_PRESSURE, LAMBDA_BETA, LAMDBA_BPOL

    def forward(self, prof_t: torch.Tensor, mp_t: torch.Tensor, **kwargs) -> List[torch.Tensor]: 
        z_enc, z_mach_enc, z_stoch_enc, mu_mach_enc, var_mach_enc, mu_stoch_enc, var_stoch_enc = self.prof2z(prof_t)
        z_cond, mu_mach_cond, var_mach_cond = self.mp2z(mp_t)
        prof_out, mp_out = self.z2prof(z_enc), self.z2mp(z_mach_enc)
        return [z_enc, z_mach_enc, z_stoch_enc, mu_mach_enc, var_mach_enc, mu_stoch_enc, var_stoch_enc, 
                z_cond, mu_mach_cond, var_mach_cond, 
                prof_out, mp_out]
    
    def prof2z(self, prof): 
        """ Encode the profile to z_stoch and z_mach"""
        enc = self.prof_encoder(prof)
        mu_mach, var_mach, mu_stoch, var_stoch = self.z_mu_mach(enc), torch.clamp(torch.nn.functional.softplus(self.z_var_mach(enc)), min=MIN_STDDEV), self.z_mu_stoch(enc), torch.clamp(torch.nn.functional.softplus(self.z_var_stoch(enc)), min=MIN_STDDEV)
        z_mach, z_stoch = self.reparameterize(mu_mach, var_mach), self.reparameterize(mu_stoch, var_stoch)
        z = torch.cat([z_mach, z_stoch], 1)
        return [z, z_mach, z_stoch, mu_mach, var_mach, mu_stoch, var_stoch]

    def mp2z(self, mp): 
        """ Conditional prior on z_mach via machine paramters """
        mu_mach_cond, var_mach_cond = self.mp_encoder(mp)
        z_cond = self.reparameterize(mu_mach_cond, var_mach_cond)
        return [z_cond, mu_mach_cond, var_mach_cond]

    def z2prof(self, z): 
        """ Decode z_mach and z_stoch to profile """
        z = self.decoder_input(z)
        dec = self.prof_decoder(z)
        prof = self.output_layer(dec)
        return prof 

    def z2mp(self, z_mach): 
        """ Decode z_mach to machine parameters """
        mp_out = self.mp_decoder(z_mach)
        return mp_out
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) ->  torch.Tensor: 
        """Reparameterization trick to sample N(mu, var) from N(0, 1)

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution 
        var : torch.Tensor
            standard deviation of the latent distribution

        Returns
        -------
        torch.Tensor
            mu + e^(var/2) + randn()
        """

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)

        z = mu + eps*std
        return z

    def inference(self, mp_in, prof_in=None, from_mean: bool = True): 
        """ DO CONDITIONAL INFERENCE! """
        # z_enc, z_mach_enc, z_stoch_enc, mu_mach_enc, var_mach_enc, mu_stoch_enc, var_stoch_enc = # self.prof2z(prof_t)
        
        z_mach_cond, mu_mach_cond, var_mach_cond = self.mp2z(mp_in)
        if prof_in is not None: 
            _, _, z_stoch_rand, _, _, _, _ = self.prof2z(prof_in)
        else: 
            z_stoch_rand = torch.normal(0, 1, size=(len(z_mach_cond), self.stoch_latent_dim))
        if not from_mean: 
            z = torch.cat([z_mach_cond, z_stoch_rand], 1)
            out_mps = self.z2mp(z_mach_cond)
        else: 
            z = torch.cat([mu_mach_cond, z_stoch_rand], 1)
            out_mps = self.z2mp(mu_mach_cond)

        out_profs = self.z2prof(z)
        return out_profs, out_mps, mu_mach_cond
    def loss_function(self, inputs, outputs, train_set=None, step=0): 
        prof_in, mp_in = inputs 
        _, _, _, mu_mach_enc, var_mach_enc, mu_stoch_enc, var_stoch_enc, _, mu_mach_cond, var_mach_cond, prof_out, mp_out = outputs 

        recon_prof, recon_mp = F.mse_loss(prof_in, prof_out), F.mse_loss(mp_in, mp_out)
        recon_loss = self.GAMMA_MP*recon_mp + self.GAMMA_PROF*recon_prof

        physics, sp_loss, beta_loss, bpol_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])

        """ 
        # kld_stoch measures the KL-div between the stochastic latent space priors given by the profile encoding against a normal distribution 
        # kld_cond measures KL-div between the machine latent space priors given by machine parameters against that given by profile encoding  
        # TODO: kld_unsup measures KL-div between mach latent space priors given by profile encoding against a normal distribution 
        """
        kld_stoch = torch.distributions.kl.kl_divergence(torch.distributions.normal.Normal(mu_stoch_enc, torch.exp(0.5*var_stoch_enc)), torch.distributions.normal.Normal(0, 1)).mean(0).sum()

        kld_mach_sup = torch.distributions.kl.kl_divergence(torch.distributions.normal.Normal(mu_mach_enc, torch.exp(0.5*var_mach_enc)), torch.distributions.normal.Normal(mu_mach_cond, torch.exp(0.5*var_mach_cond))).mean(0).sum()
        kld_mach_unsup = torch.distributions.kl.kl_divergence(torch.distributions.normal.Normal(mu_mach_enc, torch.exp(0.5*var_mach_enc)), torch.distributions.normal.Normal(0, 1)).mean(0).sum()

        # add all_kld together 
        if step % 2 == 0: 
            kld_loss = self.BETA_KLD_STOCH*kld_stoch + self.BETA_KLD_COND*kld_mach_sup
        else:
            kld_loss = self.BETA_KLD_STOCH*kld_stoch + self.BETA_KLD_MACH*kld_mach_unsup
        sp_in, beta_in, bpol_in, pressure_in = calculate_physics_constraints(prof_in, mp_in, train_set)
        sp_out, beta_out, bpol_out, pressure_out = calculate_physics_constraints(prof_out, mp_out, train_set)
        sp_loss, beta_loss, bpol_loss = F.mse_loss(sp_in, sp_out), F.mse_loss(pressure_in, pressure_out), F.mse_loss(bpol_in, bpol_out)
        physics = self.LAMBDA_PRESSURE*sp_loss + self.LAMBDA_BETA*beta_loss + self.LAMDBA_BPOL*bpol_loss
        loss = recon_loss + kld_loss + physics

        return dict(loss=loss, 
            recon_loss=recon_loss, recon_mp=recon_mp, recon_prof=recon_prof, 
            kld_loss=kld_loss, kld_cond=kld_mach_sup, kld_mach=kld_mach_unsup, kld_stoch=kld_stoch, 
            physics=physics, sp_loss=sp_loss, beta_loss=beta_loss, bpol_loss=bpol_loss)
"""
def loss_function(self, inputs, outputs, train_set=None, step=0): 
prof_in, mp_in = inputs 
_, _, _, mu_mach_enc, var_mach_enc, mu_stoch_enc, var_stoch_enc, _, mu_mach_cond, var_mach_cond, prof_out, mp_out = outputs 

recon_prof, recon_mp = F.mse_loss(prof_in, prof_out), F.mse_loss(mp_in, mp_out)
recon_loss = self.GAMMA_MP*recon_mp + self.GAMMA_PROF*recon_prof

physics, sp_loss, beta_loss, bpol_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])

""" 
# kld_stoch measures the KL-div between the stochastic latent space priors given by the profile encoding against a normal distribution 
# kld_cond measures KL-div between the machine latent space priors given by machine parameters against that given by profile encoding  
# TODO: kld_unsup measures KL-div between mach latent space priors given by profile encoding against a normal distribution 
"""
kld_stoch = torch.distributions.kl.kl_divergence(
torch.distributions.normal.Normal(mu_stoch_enc, torch.exp(0.5*var_stoch_enc)), 
torch.distributions.normal.Normal(0, 1)
).mean(0).sum()
kld_mach_sup = torch.distributions.kl.kl_divergence(
torch.distributions.normal.Normal(mu_mach_enc, torch.exp(0.5*var_mach_enc)), 
torch.distributions.normal.Normal(mu_mach_cond, torch.exp(0.5*var_mach_cond)), 
).mean(0).sum()
kld_mach_unsup = torch.distributions.kl.kl_divergence(
torch.distributions.normal.Normal(mu_mach_enc, torch.exp(0.5*var_mach_enc)), 
torch.distributions.normal.Normal(0, 1)
).mean(0).sum()

# add all_kld together 
if step % 2 == 0: 
kld_loss = self.BETA_KLD_STOCH*kld_stoch + self.BETA_KLD_COND*kld_mach_sup
else:
kld_loss = self.BETA_KLD_STOCH*kld_stoch + self.BETA_KLD_MACH*kld_mach_unsup
# add together 
if train_set is not None and self.physics: 
sp_in, beta_in, bpol_in, pressure_in = calculate_physics_constraints(prof_in, mp_in, train_set)
sp_out, beta_out, bpol_out, pressure_out = calculate_physics_constraints(prof_out, mp_out, train_set)
sp_loss, beta_loss, bpol_loss = F.mse_loss(sp_in, sp_out), F.mse_loss(pressure_in, pressure_out), F.mse_loss(bpol_in, bpol_out)
physics = self.LAMBDA_SP*sp_loss + self.LAMBDA_BETA*beta_loss + self.LAMDBA_BPOL*bpol_loss
loss = recon_loss + kld_loss + physics
return dict(loss=loss, 
    recon_loss=recon_loss, recon_mp=recon_mp, recon_prof=recon_prof, 
    kld_loss=kld_loss, kld_cond=kld_mach_sup, kld_mach=kld_mach_unsup, kld_stoch=kld_stoch, 
    physics=physics, sp_loss=sp_loss, beta_loss=beta_loss, bpol_loss=bpol_loss)
"""