# DynACD
Python implementation of Dyn_ACD algorithm described in:

- [1] Safdari H., & De Bacco C. (2023). *Community detection and anomaly prediction in dynamic networks*, <span style="color:red"> ([Communications Physics]([https://arxiv.org/abs/2404.10468](https://www.nature.com/articles/s42005-024-01889-y))).  

This is a probabilistic generative model for anomaly, community detection in temporal networks. It assigns latent variables as community memberships to nodes and flags the edges as anomalous or regular.  <br>

If you use this code please cite [1].   

Copyright (c) 2023 [Hadiseh Safdari](https://github.com/hds-safdari), and [Caterina De Bacco](http://cdebacco.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's Included

- **`Dyn_ACD`**: Contains the following:
  - `Dyn_ACD_wtemp.py`: Latest version of the Dyn_ACD algorithm.
  - Notebook: `Dyn_ACD_Inf_debug-wtemp.ipynb` (for debugging the Dyn_ACD algorithm).
  - `Dyn_ACD_gm.py`: Latest version of the generative algorithm for benchmark synthetic data.
  - Notebook: `Dyn_ACD_Gm_debug.ipynb` (for debugging the generative algorithm).
  - `AnomalyInjection.py`: To inject anomalous edges in Real-World datasets.
  
  - Notebook: `analyze_transfermarkt.ipynb` (to analyse the transfermarkt dataset applying Dyn_ACD algorithm). 
- **`data/input`**: Contains:
  - An example of a network with an intrinsic community structure and given anomaly coefficients.
  - Example files to initialize latent variables (synthetic data).

- **`data/output`**: Contains:
  - Sample results to test the code.
  
  

## Requirements
The project has been developed using Python 3.7 with the packages contained in *requirements.txt*. It is possible to install the dependencies using pip:
`pip install -r requirements.txt`

## Test
You can run tests to reproduce results contained in `data/output` by running (inside `code` directory):  

```bash 
python -m unittest main_cv_w_temp.py
```

## Usage
To test the program on the given example file, type:  

```bash
cd code
python main.py
```

It will use the sample network contained in `./data/input/synthetic/200_08`. The adjacency matrix *syn_200_2_8.0_8_0.2_2_2_0.1_9.dat* represents a  weighted network with **N=$200$** nodes, average degree **$8$**, total time **$T=8$**, **$\pi=0.2$**, **$ell=0.2$**, **$\beta=0.2$**, **$\rho=0.1$**, **K=$2$**,  equal-size unmixed communities with an **assortative** structure. 

### Parameters 
- **-a** : Different variations of the algorithm, *(default=ACD_Wdynamic)*.
- **-K** : Number of communities, *(default=3)*.
- **-A** : Input file name of the adjacency matrix.
- **-f** : Path of the input folder, *(default='../data/input/')*.
- **-o** : Path of the input folder, *(default='../data/output/')*.
- **-e** : Name of the source of the edge, *(default='source')*.
- **-t** : Name of the target of the edge, *(default='target')*.
- **-d** : Flag to force a dense transformation of the adjacency matrix, *(default=False)*.
- **-E** : Flag to run the algorithm with anomaly  or without anomaly detection approach, *(default=0)*.


You can find a list by running (inside `code` directory): 

```bash
python main.py --help
```

## Input format
The network should be stored in a *.dat* file. An example of rows is

`node1 node2 3` <br>
`node1 node3 1`

where the first and second columns are the _source_ and _target_ nodes of the edge, respectively; the third column tells if there is an edge and the weight. In this example the edge node1 --> node2 exists with weight 3, and the edge node1 --> node3 exists with weight 1.

Other configuration settings can be set by modifying the *setting\_\*_.yaml* files: 

- *setting\_syn_data.yaml* : contains the setting to generate synthetic data
- *setting\_CRep.yaml* : contains the setting to run the algorithm Dyn_ACD, while keeping affinity matrix dynamic
- *setting\_setting_ACD_Wdynamic.yaml* : contains the setting to run the algorithm Dyn_ACD, while keeping affinity matrix static

## Output
The algorithm returns a compressed file inside the `data/output` folder. To load and print the out-going membership matrix:

```bash
import numpy as np 
theta = np.load('theta_acd.npz')
print(theta['u'])
```

_theta_ contains the two NxK membership matrices **u** *('u')* and **v** *('v')*, the 1xKxK (or 1xK if assortative=True) affinity tensor **w** *('w')*, the anomaly coefficients **$\pi$**, **$\mu$**,  **$\ell$**, the disappearance rate of regualr, and anomalous edges, respectively,  **$\beta$**, **$\gamma$**, the total number of iterations *('max_it')*, the value of the maximum pseudo log-likelihood *('maxPSL')* and the nodes of the network *('nodes')*.  

<!-- For an example `jupyter notebook` importing the data, see `code/analyse_results.ipynb`. -->

