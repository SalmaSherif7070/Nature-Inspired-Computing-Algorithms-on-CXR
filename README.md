# Nature-Inspired Computation (NIC) — Final Project

## Overview
This project applies **Nature-Inspired Computation (NIC)** techniques to **optimize deep learning models for COVID-19 detection from chest X-ray images**.  
The work integrates **Deep Learning**, **Metaheuristic Optimization**, and **Explainable AI (XAI)** to produce a **highly accurate, robust, and interpretable medical imaging model**.

Two deep learning models (**DCSNN** and **SCOVNET**) are optimized using multiple metaheuristic algorithms, and their predictions are further enhanced with explainability techniques suitable for medical decision support.

---

## Dataset
The dataset used in this project is publicly available on Kaggle:

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/salmasherif202200622/feature-selelcted-dataset

### Dataset Description
- Chest X-ray images for **three classes**:
  - COVID-19
  - Pneumonia
  - Normal
- Constructed by merging datasets from **Kaggle** and **GitHub**
- Contains **≥ 7,000 samples**
- Includes preprocessing and feature-selection-ready representations

The same dataset is used across all phases to ensure **fair comparison and consistent evaluation**.

---

## Project Objectives
- Build deep learning models for **COVID-19, Pneumonia, and Normal classification**.
- Optimize model performance using **nature-inspired metaheuristic algorithms**.
- Perform **feature selection using Ant Colony Optimization (ACO)**.
- Apply **meta-optimization**, where one metaheuristic optimizes another.
- Enhance model interpretability using **Explainable AI (XAI)**.
- Deliver a **final optimized and explainable medical AI model**.


---

## Models Used
- **DCSNN** (Deep Convolutional Neural Neural Network)
- **SCOVNET** (CNN-based architecture for COVID-19 detection)

Both models are evaluated before and after optimization.

---

## Preprocessing and Feature Handling
- Dataset merging and exploratory data analysis (EDA)
- Normalization and standardization
- Gaussian smoothing
- Histogram equalization
- Gamma correction
- Oversampling to handle class imbalance
- Removal of annotated, overexposed, and underexposed X-rays
- Feature selection using **Ant Colony Optimization**

---

## Metaheuristic Algorithms Applied
Across different stages, the following algorithms are used:

- Hill Climbing
- Tabu Search
- Simulated Annealing
- Ant Colony Optimization
- Particle Swarm Optimization (PSO)
- Gray Wolf Optimization

Hybrid and meta-optimized approaches are also applied.

---

## Mandatory Steps & Deliverables

### 1. Model Parameter Optimization
- Optimize **DCSNN** and **SCOVNET** hyperparameters using:
  - Hill Climbing
  - Tabu Search
  - Simulated Annealing
  - Ant Colony Optimization
  - Particle Swarm Optimization
  - Gray Wolf Optimization
- Compare training and validation performance.

---

### 2. Feature Selection
- Apply **Ant Colony Optimization (ACO)** to:
  - Reduce feature dimensionality
  - Eliminate redundant features
  - Improve generalization

---

### 3. Metaheuristic Parameter Optimization
- Select metaheuristics with tunable parameters (e.g., PSO).
- Use another metaheuristic (e.g., Hill Climbing) to optimize:
  - PSO parameters (C1, C2, inertia weight)
- Evaluate the impact on convergence and accuracy.

---

### 4. Explainability Optimization
Apply **four metaheuristic algorithms** to optimize XAI methods:

- **Grad-CAM**
- **LIME**
- **SHAP**
- **DeepLIFT**
- **Permutation Importance**

Optimization focuses on:
- Explanation stability
- Noise reduction
- Clinical relevance of highlighted regions

---

## Phase Deliverables

### Phase 1
- Dataset description and justification
- Preprocessing and feature handling
- Baseline DCSNN and SCOVNET models
- Optimization stage I results
- Short presentation

### Phase 2
- Fully optimized model configurations
- Optimization stage II and hybrid approaches
- XAI visualizations and analysis
- Full technical report
- Clean GitHub repository
- Final presentation

---

## How to Run the Project

1. Download the dataset from Kaggle and place it in the `assets/` directory.
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
- Use the **same dataset** throughout all experiments.
- Fix random seeds for reproducibility.
- Report computation time for each metaheuristic.
- Evaluation metrics include:
  - Accuracy
  - Precision, Recall, F1-score
  - Loss
  - Execution time
  - XAI stability and consistency

---

## References
- Project brief and grading rubric
- Ant Colony Optimization for feature selection (as provided in project documentation)

---

## Contributing
- Add notebooks and scripts to the `src/` directory.
- Dataset source:  
  https://www.kaggle.com/datasets/salmasherif202200622/feature-selelcted-dataset
