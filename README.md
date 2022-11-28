# psi_2022

We study VAEs and how they can be used to compress information from diagnostic data into a representation of the plasma state.

To view an overview of what is capable, see `POSTER.pdf`, a poster from the 25th International Conference on Plasma Surface Interactions in Controlled Fusion Devices (PSI-25).

## Building data

In order to gather data to reproduce the results, you need a JET account. If you have access to JET and Heimdall/NoMachine, then see `/src/data/README.md` for more information on bulding the dataset.

## Usage & Installation

If you were able to download the above data, then you can move to installing this package. 

Assuming you have some virtual environement and have already cloned this repository:

0. `cd` into the cloned directory.  
1. To train the model, run `python3 train.py`. This will produce a file `./{model_name}.pth`
2. You can use this model file to plot in `plotting.py`

Feel free to play with the hyperparameters in `train.py`

## Citation

JNME Submission TBD.

* [arxiv Paper](https://arxiv.org/abs/2208.00206)
``` 
@misc{https://doi.org/10.48550/arxiv.2208.00206,
    doi = {10.48550/ARXIV.2208.00206},
    author = {Kit, A. and Jaervinen, A. and Wiesen, S. and Poels, Y. and Frassinetti, L.},
    keywords = {Plasma Physics (physics.plasm-ph), FOS: Physical sciences, FOS: Physical sciences},
    title = {Developing Deep Learning Algorithms for Inferring Upstream Separatrix Density at JET},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
    }
```

