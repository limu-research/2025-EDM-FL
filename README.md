# Overview

This repository contains supplementary materials for the following conference paper:

Shunsuke Yoneda, Valdemar Švábenský, Gen Li, Daisuke Deguchi, and Atsushi Shimada.\
**Ranking-Based At-Risk Student Prediction Using Federated Learning and Differential Features**.\
In Proceedings of the 18th International Conference on Educational Data Mining ([EDM 2025](https://educationaldatamining.org/edm2025/)).

```bibtex
@inproceedings{Yoneda2025ranking,
    author    = {Yoneda, Shunsuke and \v{S}v\'{a}bensk\'{y}, Valdemar and Li, Gen and Deguchi, Daisuke and Shimada, Atsushi},
    title     = {{Ranking-Based At-Risk Student Prediction Using Federated Learning and Differential Features}},
    booktitle = {Proceedings of the 18th International Conference on Educational Data Mining},
    series    = {EDM '25},
    location  = {Palermo, Italy},
    editor    = {},
    publisher = {International Educational Data Mining Society},
    month     = {07},
    year      = {2025},
    pages     = {},
    numpages  = {14},
    url       = {},
    doi       = {},
}
```

# Preprocessing with E2Vec
We use [E2Vec](https://github.com/limu-research/2024-edm-e2vec) for preprocessing student learning log data.  

# File Description

## Main Notebook

- `build_model_proposed_method.ipynb`  
  This file builds and trains the proposed model based on federated learning with differential features.

## Evaluation Scripts

- `evaluate_model/`

  - `evaluation_proposed_method.py`  
    This file generates the results shown in **Table 4 (Proposed Method)** and **Table 5 (Proposed Method)** using the full proposed method.

  - `evaluation_not_federated.py`  
    This file generates the results shown in **Table 4 (Baseline Method 1)** using centralized (non-federated) learning.
    
  - `evaluation_not_difference.py`  
    This file generates the results shown in **Table 5 (Baseline Method 2)** without using differential features.

  - `evaluation_proposed_method_early.py`  
    This file generates the results shown in **Table 6 (Proposed Method)**, which evaluates the proposed method on early-stage prediction.

  - `evaluation_relationship_lecture_sessions.py`  
    This file generates the results shown in **Figure 10–14**, showing prediction performance over lecture sessions.

  - `evaluation_relationship_grades_and_rankings.py`  
    This file generates the results shown in **Figure 15–19**, showing the relationship between At-risk rankings and actual grades.
