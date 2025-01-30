# Interpreting Temporal Graph Neural Networks with Koopman Theory (2024) 
<!--- ({Venue} {Year})

[![ICLR](https://img.shields.io/badge/{Venue}-{Year}-blue.svg?)]({Link to paper page})
[![paper](https://custom-icon-badges.demolab.com/badge/paper-pdf-green.svg?logo=file-text&logoSource=feather&logoColor=white)]({Link to paper page})
-->

<!---
[![poster](https://custom-icon-badges.demolab.com/badge/poster-pdf-orange.svg?logo=note&logoSource=feather&logoColor=white)]({Link to the poster/presentation})
[![arXiv](https://img.shields.io/badge/arXiv-{Arxiv.ID}-b31b1b.svg?)]({Link to Arixv})
-->
[![arXiv](https://img.shields.io/badge/arXiv-2410.13469-b31b1b.svg?)](https://arxiv.org/pdf/2410.13469)


This repository contains the code for the reproducibility of the experiments presented in the paper "Interpreting Temporal Graph Neural Networks with Koopman Theory" (2024).
<!--({Venue} {Year}). --> 
We present a novel approach to interpret temporal graph models using Koopman theory, combining DMD and SINDy to uncover key spatial and temporal patterns.

**Authors**: [Michele Guerra](https://en.uit.no/ansatte/person?p_document_id=767125&p_dimension_id=88140), [Simone Scardapane](https://www.sscardapane.it/), [Filippo Maria Bianchi](https://sites.google.com/view/filippombianchi/home)

---

## In a nutshell

Spatiotemporal graph neural networks (STGNNs) have shown promising results in many domains, from forecasting to epidemiology. However, understanding the dynamics learned by these models and explaining their behaviour is significantly more complex than for models dealing with static data. 
Inspired by Koopman theory, which allows a simpler description of intricate, nonlinear dynamical systems, we introduce an explainability approach for temporal graphs. We present two methods to interpret the STGNN’s decision process and identify the most relevant spatial and temporal patterns in the input for the task at hand. 
The first relies on dynamic mode decomposition (DMD), a Koopman-inspired dimensionality reduction method. The second relies on sparse identification of nonlinear dynamics (SINDy), a popular method for discovering governing equations, which we use for the first time as a general tool for explainability. 
We show how our methods can correctly identify interpretable features such as infection times and infected nodes in the context of dissemination processes.

<p align=center>
	<img src="./images/koopman.gif" alt="Example of spatiotemporal explanation."/>
</p>

---

## Directory structure

The directory is structured as follows:

```
.
├── configs/
├── dataset/
├── images/
├── koopman/
├── licenses/
├── models/
│   └── saved
├── utils/
├── requirements.txt
├── train_tsl_model.py
└── experiment_graph.py

```


## Datasets

All datasets are stored in the folder `dataset`.
Each dataset comes with a README file that provides sources and a description.


## Configuration files

The `configs` directory stores all the configuration files used to run the experiment.

## Requirements

We run all the experiments in `python 3.10`. To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda create --name env_name --file requirements.txt
conda activate env_name
```

Alternatively, you can install all the requirements listed in `requirements.txt` with pip:

```bash
pip install -r requirements.txt
```

## Experiments

The scripts used for the experiments in the paper are in the main folder.

* `train_tsl_model.py` is used to train the TGNN using the configurations contained in the config file. An example of usage is

	```
	python train_tsl_model.py
	```
* `experiment_graph.py` is used to perform the explainability methods using the hyperparamenters set in the config file. Use it as follows

	```
	python experiment_graph.py
	```


## Third-Party licenses

This project incorporates code from other open-source projects, all under the MIT License. For more details, see the `licenses/` directory for full license texts.

- [Torch Spatiotemporal](https://github.com/TorchSpatiotemporal/tsl): File `models/DynGraphConvRNN.py` is a modification of code from the cited library, to allow time-varying topology.
- [SindyAutoencoders](https://github.com/kpchamp/SindyAutoencoders): Some part of file `koopman\sindy.py` are taken from the cited library.
- [KANN](https://github.com/azencot-group/KANN): File `koopman\dmd.py` is partly based on the cited library.


## Bibtex reference

If you find this code useful please consider citing our paper:

```bibtex
@misc{guerra2024interpretingtemporalgraphneural,
      title={Interpreting Temporal Graph Neural Networks with Koopman Theory}, 
      author={Michele Guerra and Simone Scardapane and Filippo Maria Bianchi},
      year={2024},
      eprint={2410.13469},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.13469}, 
}
```
