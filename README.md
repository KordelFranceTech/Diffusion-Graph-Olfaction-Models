---
language: 
  - en
tags:
- embeddings
- multimodal
- olfaction-vision-language
- olfaction
- olfactory
- scentience
- vision-language
- vision
- language
- robotics
license: mit
datasets:
- kordelfrance/olfaction-vision-language-dataset
- detection-datasets/coco
base_model: Scentience-OVL-Embeddings-Base
---

Diffusion Graph Neural Networks for Robust Olfactory Navigation in Robotics
----

<div align="center">

**Olfaction • Vision • Language**


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow?logo=google-colab)](https://colab.research.google.com/drive/1z-ITTEfVtMMbfbN50u2AfQhzvuYkrRn7?usp=sharing)
[![Paper](https://img.shields.io/badge/Research-Paper-red)](https://arxiv.org/abs/2506.00455)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces)

</div>


An open-sourced diffusion-based graph neural network for olfaction-vision-language tasks.

---

## Model Description

Navigation by scent is a capability in robotic systems that is rising in demand. 
However, current methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. 
To address challenges in olfactory navigation, we introduce a novel machine learning method using diffusion-based molecular gen-
eration that can be used by itself or with automated olfactory
dataset construction pipelines. This generative process of our diffusion model expands the chemical space beyond the limitations
of both current olfactory datasets and training methods, enabling
the identification of potential odourant molecules not previously
documented. The generated molecules can then be more accurately validated using advanced olfactory sensors, enabling
them to detect more compounds and inform better hardware
design. By integrating visual analysis, language processing, and
molecular generation, our framework enhances the ability of
olfaction-vision models on robots to accurately associate odours
with their correct sources, thereby improving navigation and
decision-making through better sensor selection for a target
compound in critical applications such as explosives detection,
narcotics screening, and search and rescue. Our methodology
represents a foundational advancement in the field of artificial
olfaction, offering a scalable solution to challenges posed by
limited olfactory data and sensor ambiguities.

For the full training set, please see the fully open-source dataset [here on HuggingFace](https://huggingface.co/datasets/kordelfrance/olfaction-vision-language-dataset).

---

## Models
We offer two models with this repository:
 - (1) `constrained`: A diffusion model with its associated olfactory conditioner that is constrained to only generate molecules based on the atoms `C`, `N`, `O`, `F`, `P`, `S`, and `Cl`.
 - (2) `unconstrained`: A diffusion model with its associated olfactory conditioner that is unconstrained and may generate molecules from any atom.


## Directory Structure

```text
DiffusionGraphOlfactionModels/
├── data/                     # Example dataset
├── src/                      # Model training and inferenct tools
├── notebooks/                # Colab-ready notebooks
├── models/                   # Pre-trained models for immediate use
├── requirements.txt          # Python dependencies
├── LICENSE                   # Licensing terms of this repository
└── README.md                 # Overview of repository contributions and usage
```

---

## Getting Started

The easiest way to get started is to open the Colab notebook and begin there.
To explore the model and train locally, follow the steps below:

#### 1. Clone the Repository

```bash
git clone https://github.com/KordelFranceTech/Diffusion-Graph-Olfaction-Models.git
cd DiffusionGraphOlfactionModels
````

#### 2. Create a Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run Inference or Train Models
Run inference:
```bash
python scripts/main.py
```
Train Models:
```bash
jupyter notebook notebooks/Olfaction_Diffusion-Train.ipynb
```

---

## Citation

If you use these models in your research, please cite as follows:

```bibtex
@misc{france2025diffusiongraphneuralnetworks,
      title={Diffusion Graph Neural Networks for Robustness in Olfaction Sensors and Datasets}, 
      author={Kordel K. France and Ovidiu Daescu},
      year={2025},
      eprint={2506.00455},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00455}, 
}
```

---


## License

This dataset is released under the [MIT License](https://opensource.org/license/mit).
