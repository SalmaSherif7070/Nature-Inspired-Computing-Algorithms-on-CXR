# Nature-Inspired Computation (NIC) — Final Project

## Overview
This project integrates **Nature-Inspired Computation (NIC)**, **Deep Learning**, and **Explainable AI (XAI)** to develop a **fully optimized and interpretable deep learning model**.  
Multiple **metaheuristic optimization algorithms** are applied across model training, feature selection, and explainability using a **single high-dimensional dataset** containing **at least 7,000 samples**.

The final outcome is a **robust, optimized, and explainable deep model**, supported by quantitative evaluation and visual explanations.

---

## Dataset
The dataset used in this project is publicly available on Kaggle:

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/salmasherif202200622/feature-selelcted-dataset

The dataset is used consistently across all phases of the project to ensure fair comparison and reliable evaluation.

---

## Project Objectives
- Apply nature-inspired metaheuristic algorithms to optimize deep learning models.
- Perform feature selection using **Ant Colony Optimization (ACO)**.
- Optimize metaheuristic algorithm parameters using another metaheuristic (meta-optimization).
- Enhance and optimize **XAI methods** (SHAP, LIME, Grad-CAM).
- Deliver a final optimized model with reliable performance and explainability.

---

## Project Structure
assets/
│── Dataset and supporting assets

src/
│── Phase 1.1.ipynb
│── Phase 1.2.ipynb
│── Phase 2.1.ipynb
│── Phase 2.2.ipynb
│── Phase 2.3.ipynb
│── Phase 2.4.ipynb


---

## Key Requirements
- Use **one dataset only** (image or text — no tabular data).
- Dataset size must be **≥ 7,000 samples**.
- Apply **7–9 unique metaheuristic algorithms** across all phases.
- Produce:
  - One final optimized deep learning model.
  - Explainable AI visualizations with quality and stability analysis.

---

## Mandatory Steps & Deliverables

### 1. Model Parameter Optimization
- Optimize deep learning hyperparameters using **at least 6 metaheuristic algorithms**.
- Compare results against baseline models.

### 2. Feature Selection
- Apply **Ant Colony Optimization (ACO)** to select the most informative features and reduce redundancy.

### 3. Metaheuristic Parameter Optimization
- Select **two metaheuristic algorithms** with tunable parameters (e.g., PSO, GA).
- Use **one metaheuristic** to optimize their internal parameters (e.g., Hill Climbing optimizing PSO parameters).

### 4. Explainability Optimization
- Apply **four metaheuristic algorithms** to optimize XAI methods:
  - SHAP
  - LIME
  - Grad-CAM

---

## Phase Deliverables

### Phase 1
- Dataset description and justification
- Data preprocessing and exploration
- Baseline deep learning model
- Comparative analysis of optimization results
- Short presentation

### Phase 2
- Fully optimized model results
- XAI visualizations and evaluation
- Full technical report
- Clean GitHub repository (code + documentation)
- Final presentation

---

## How to Run the Project

1. Download the dataset from Kaggle and place it inside the `assets/` directory.
2. Update dataset paths in all notebooks.
3. Create a Python environment and install dependencies:
   - TensorFlow / PyTorch
   - scikit-image
   - opencv-python
   - shap
   - lime
   - seaborn
   - matplotlib
4. Run notebooks sequentially:
   - `Phase 1.1.ipynb` → `Phase 1.2.ipynb`
   - Then all Phase 2 notebooks in order.

---

## Notes & Best Practices
- Use the **same dataset** throughout all phases.
- Fix and report random seeds for reproducibility.
- Record computation time for each optimization algorithm.
- Report evaluation metrics consistently:
  - Accuracy
  - Precision, Recall, F1-score
  - Loss
  - Execution time
  - XAI stability and consistency metrics

---

## References
- Follow the official project brief for constraints and grading criteria.
- Ant Colony Optimization for feature selection: refer to the provided project documentation and references.

---

## Contributing
- Add all notebooks and scripts to the `src/` directory.
- Place datasets [here](https://www.kaggle.com/datasets/salmasherif202200622/feature-selelcted-dataset/)
