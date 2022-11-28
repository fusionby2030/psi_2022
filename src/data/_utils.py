import numpy as np 
import torch 

def initialize_dataloaders(dataset, val_split: float = 0.3, test_split: float = 0.1, batch_size: int = 256, nw: int = 4): 
    pulse_idxs = np.arange(dataset.total_num_pulses)

    make_vals = True
    # Need to randomly shuffle the pulses, as they are sorted originally from 30000 -> onward!
    np.random.shuffle(pulse_idxs)
    num_val_pulses = int(val_split * dataset.total_num_pulses)
    num_test_pulses = int(test_split * dataset.total_num_pulses)
    train_pulse_idxs = pulse_idxs[:-(num_val_pulses  + num_test_pulses)]
    val_pulse_idxs = pulse_idxs[-(num_val_pulses  + num_test_pulses):-num_test_pulses]
    test_pulse_idxs = pulse_idxs[-num_test_pulses:]
    train_norms = dataset.get_norms(train_pulse_idxs, return_norms=True)
    
    # Need to then get the relevant slices from the list of pulses...
    train_slice_idxs, val_slice_idxs, test_slice_idxs = [], [], []
    for set_pulse_idxs, set_slice_idxs in zip([train_pulse_idxs, val_pulse_idxs, test_pulse_idxs], [train_slice_idxs, val_slice_idxs, test_slice_idxs]): 
        for num in set_pulse_idxs: 
            if num == 0: 
                start_slice_idx = 0
            else: 
                start_slice_idx = dataset.cumsum_num_slices[num-1] + 1 # the plus one comes from the cumsum_num_slices subtracting one at the begninggn
            end_slice_idx = dataset.cumsum_num_slices[num] + 1
            set_slice_idxs.extend(np.arange(start_slice_idx, end_slice_idx))

    dataset.train_slice_idxs = train_slice_idxs
    dataset.val_slice_idxs = train_slice_idxs
    dataset.test_slice_idxs = train_slice_idxs

    train_dataset = torch.utils.data.Subset(dataset, train_slice_idxs)
    val_dataset = torch.utils.data.Subset(dataset, val_slice_idxs)
    test_dataset = torch.utils.data.Subset(dataset, test_slice_idxs)

    pin_memory = True 
        
    dataloader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw, 
            pin_memory=pin_memory
        )
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    valid_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
    return train_loader, valid_loader, test_loader

def get_dataloaders(dataset, batch_size: int = 256):
    train_dataset = torch.utils.data.Subset(dataset, dataset.train_slice_idxs)
    val_dataset = torch.utils.data.Subset(dataset, dataset.val_slice_idxs)
    test_dataset = torch.utils.data.Subset(dataset, dataset.test_slice_idxs)

    dataloader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
        )
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    valid_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
    return train_loader, valid_loader, test_loader
