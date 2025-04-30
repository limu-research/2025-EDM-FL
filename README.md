# Overview

This repository contains the implementation and evaluation code for our proposed method for at-risk student prediction using federated learning and differential features.

# File Description

## Main Notebook

- `build_model_proposed_method.ipynb`  
  Builds and trains the proposed model based on federated learning with differential features.

## Evaluation Scripts

- `evaluate_model/`

  - `evaluation_proposed_method.py`  
    Generates the results shown in **Table 4** and **Table 5** using the full proposed method.

  - `evaluation_not_federated.py`  
    Generates the results shown in **Table 4** using centralized (non-federated) learning (Baseline Method 1).
    
  - `evaluation_not_difference.py`  
    Generates the results shown in **Table 5** without using differential features (Baseline Method 2).

  - `evaluation_proposed_method_early.py`  
    Generates the results shown in **Table 6**, which evaluates the proposed method on early-stage prediction.


  - `evaluation_relationship_lecture_sessions.py`  
    Generates the results shown in **Figure 10–14**, showing prediction performance over lecture sessions.

  - `evaluation_relationship_grades_and_rankings.py`  
    Generates the results shown in **Figure 15–20**, showing the relationship between At-risk rankings and actual grades.
