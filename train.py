from src.models import DIVA
from src.data import MemMapDataset_O
from src.data._utils import initialize_dataloaders
from src.common.utils import save_model
from typing import List 
import os 
import timeit
import torch 
import numpy as np
from tqdm import tqdm 

""" Training Paramters """
EPOCHS: int = 100 # Around 50 it starts getting okay 
BATCH_SIZE: int = 512 # 512
LR: float = 0.01 # 0.01

""" Model Paramters """
MACH_LATENT_DIM:int = 10 # 8
STOCH_LATENT_DIM:int = 3  # 3
CONV_FILTER_SIZES: List[int] = [2, 4, 6]  # [2, 4, 6]
MP_REG_LAYER_WIDTH: int = 20 # 40
MP_REG_LAYER_DEPTH: int = 4 # 6
MP_REG_LAYER_SIZES: List[int] = [MP_REG_LAYER_WIDTH]*MP_REG_LAYER_DEPTH 

""" 
DIVA Specific KL-divergences
_STOCH : applied to Z_stoch
_COND_SUP : applied to mach_cond vs mach_enc 
_COND_UNSUP : applied to mach_enc when unsupervised 
"""
BETA_STOCH: float = 0.5 # 0.8
BETA_COND_SUP: float = 1.0 # 1.0
BETA_COND_UNSUP: float = 0.001 # 0.0086

""" 
Loss function parameterization 
GAMMA : applied to reconstructions of profiles and machine parameters
LAMBDA : applied to the physics losses
"""
PHYSICS: bool = True
GAMMA_PROF: float = 1200.0  # 550
GAMMA_MP: float = 20.0  # 2.6618
LAMBDA_PRESSURE: float = 500.0 # 55.0
LAMBDA_BETA: float= 1000.0 # 10.0
LAMDBA_BPOL: float = 100.0 # 31.0

""" Scheduler paramterizations """
SCHEDULER_STEP_SIZE: int = 10 # 10
SCHEDULER_GAMMA: int = 0.9 # 0.99
""" Save Information """
MODEL_NAME = 'DIVA'
# ['BTF', 'D_tot', 'N_tot', 'IpiFP', 'PNBI_TOT', 'PICR_TOT', 'PECR_TOT',  'k', 'delRoben', 'delRuntn', 'ahor', 'Rgeo', 'q95', 'Vol']

hparams = dict(mach_latent_dim=MACH_LATENT_DIM, stoch_latent_dim=STOCH_LATENT_DIM, 
                        conv_filter_sizes=CONV_FILTER_SIZES, mp_layer_sizes=MP_REG_LAYER_SIZES, 
                        GAMMA_PROF=GAMMA_PROF, GAMMA_MP=GAMMA_MP, 
                        BETA_KLD_COND=BETA_COND_SUP, BETA_KLD_STOCH=BETA_STOCH, BETA_KLD_MACH=BETA_COND_UNSUP, 
                        LAMBDA_PRESSURE=LAMBDA_PRESSURE, LAMBDA_BETA=LAMBDA_BETA, LAMDBA_BPOL=LAMDBA_BPOL, 
                        physics=PHYSICS, model_name=MODEL_NAME)


def main(base_data_path: str = None): 
    dataset = MemMapDataset_O(data_path=base_data_path)

    hparams['out_length'] = dataset.prof_length 
    hparams['action_dim'] = dataset.mp_dim 

    model = DIVA(**hparams)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(9), lr=LR)
    train_dl, val_dl, test_dl = initialize_dataloaders(dataset, batch_size=BATCH_SIZE)

    training_loss, best_val_loss = [], np.inf
    # epoch_iter = tqdm(range(EPOCHS))
    epoch_iter = range(EPOCHS)
    for epoch in epoch_iter:
        epoch_loss, epoch_loss_dict = train_epoch(model, optimizer, train_dl, dataset)
        training_loss.append(epoch_loss)
        if epoch % 5 == 0: 
            val_loss, _ = test_epoch(model, val_dl, dataset)
            if val_loss < best_val_loss: 
                print(f'Epoch: {epoch}\nNew best val loss: {val_loss:.4}, saving model')
                save_model(model, hparams, dataset)

def train_epoch(model, optimizer, loader, dataset): 
    epoch_loss = 0.0
    for steps, batch in enumerate(loader): 
        batch_profs, batch_mps = batch 
        preds = model.forward(batch_profs, batch_mps)
        
        loss_dict = model.loss_function(batch, preds, dataset, step=steps)

        loss = loss_dict.pop('loss') / len(loader)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item() 
    return epoch_loss, loss_dict

def test_epoch(model, loader, dataset):
    epoch_loss = 0.0
    with torch.no_grad(): 
        for steps, batch in enumerate(loader): 
            batch_profs, batch_mps = batch 
            preds = model.forward(batch_profs, batch_mps)
            loss_dict = model.loss_function(batch, preds, dataset, step=steps)
            loss = loss_dict.pop('loss') / len(loader)
            epoch_loss += loss.item() 
    return epoch_loss, loss_dict

if __name__ == '__main__': 
    data_path: str = os.path.join(os.getenv('PROC_DIR'), 'JET_PDB_PULSE_ARRAYS') 
    main(data_path) 