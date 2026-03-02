# Master Thesis: Advanced Component-wise Gradient Boosting for Improved Generalizability

This repository contains the code and environment required to reproduce the experiments and figures for this thesis.

## Reproducing Results

Follow these steps to reproduce the experiments and figures presented in this project.

### 1. Installation
To set up the environment and install all necessary dependencies, run:

bash
pip install -r requirements.txt


### 2. Run Full Experiments
To execute the complete experimental pipeline from scratch (this will generate the results from scratch):

bash
python run_experiments.py


### 3. Generate Figures & Post-processing
If you want to skip the experiments and only reproduce all figures using the pre-existing data in `results.csv`, run the following scripts **in this specific order**:

1. **Averages:** bash
   python postprocess_results_avg.py
   
2. **Statistical Tests:** bash
   python postprocess_results_tests.py
   
3. **Significance Heatmap:** bash
   python postprocess_results_heatmap.py
   
4. **Drift Heatmap:** bash
   python postprocess_results_heatmap2.py
   
5. **Feature Selection Plots:** bash
   python postprocess_results_feature_selection.py