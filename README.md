# Overview

This repository contains supplementary materials for the following conference paper:

Shunsuke Yoneda, Valdemar Švábenský, Gen Li, Daisuke Deguch, and Atsushi Shimada.\
**Ranking-Based At-Risk Student Prediction Using Federated Learning and Differential Features**.\
In Proceedings of the 18th International Conference on Educational Data Mining ([EDM 2025](https://educationaldatamining.org/edm2025/)).

## Preprocessing with E2Vec
We use [E2Vec](https://github.com/limu-research/2024-edm-e2vec) for preprocessing student learning log data.  

# File Description

## Main Notebook

- `build_model_proposed_method.ipynb`  
  This file builds and trains the proposed model based on federated learning with differential features.

## Evaluation Scripts

- `evaluate_model/`

  - `evaluation_proposed_method.py`  
    This file generates the results shown in **Table 4(Proposed Method)** and **Table 5(Proposed Method)** using the full proposed method.

  - `evaluation_not_federated.py`  
    Generates the results shown in **Table 4(Baseline Method 1)** using centralized (non-federated) learning.
    
  - `evaluation_not_difference.py`  
    Generates the results shown in **Table 5(Baseline Method 2)** without using differential features .

  - `evaluation_proposed_method_early.py`  
    Generates the results shown in **Table 6(Proposed Method)**, which evaluates the proposed method on early-stage prediction.


  - `evaluation_relationship_lecture_sessions.py`  
    Generates the results shown in **Figure 10–14**, showing prediction performance over lecture sessions.

  - `evaluation_relationship_grades_and_rankings.py`  
    Generates the results shown in **Figure 15–20**, showing the relationship between At-risk rankings and actual grades.
