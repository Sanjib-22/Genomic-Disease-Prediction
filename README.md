# Genomic Disease Prediction using Polygenic Risk Scores (PRS)

The project mainly focuses on evaluating machine learning models and predicting specific genetically inherited diseases using **Polygenic Risk Scores (PRS)**, leveraging Genomic data and machine learning techniques.
The work was carried out as a part of an internship at **IIT Guwahati** .

## Project Overview

The goal of this project is to assess the predicitive capability of **common SNPs (Single Nucleotide Polymorphism)** using PRS, applied across multiple diseases:
- **Rheumatoid Arthritis**
- **Alzheimer's Disease**
- **Type 2 Diabetes**
- **Hyperthyroidism**

## Project Structure

```
Internship_Project/
├─ Final Datasets/
│  ├─ ALZ_final/
│  ├─ HYP_final/
│  ├─ RA_final/
│  └─ T2D_final/
├─ Models/
│  ├─ Alzheimers.ipynb
│  ├─ Hyperthyroidism.ipynb
│  ├─ Rheumatoid_Arhtiritis.ipynb
│  └─ Type_2_diabetes.ipynb
├─ Outputs/
│  ├─ ALZ_results/
│  ├─ HYP_results/
│  ├─ RA_results/
│  ├─ T2D_results/
│  └─ Table.ipynb
├─ Dataset_preprocessing/  
│  ├─ Dataset_processing_ALZ.ipynb
│  ├─ Dataset_processing_HYP.ipynb
│  ├─ Dataset_processing_RA.ipynb
│  └─ Dataset_processing_T2D.ipynb
├─ .gitignore
├─ environment.yaml
├─ LICENSE
└─ README.md
```

## Datasets Used

- **1000 Genomes Project** (genotype data)
- Phenotype labels were **simulated** to follow disease-specific prevalance are used for binary classificaton.
- Two splits were tested:
  - **50/50 Case-Control**
  - **10/90 Case-Control** *(produced more realistic and better-performing results)*

## Methodology

- **Dataset Collection**:
  - Used **GWAS Catalog** for SNP collection for the four diseases and **1000 Genomes Project** for chromosomes related to each particular trait
  - Used **pandas** to remove the NULL values and taking only the necessary columns *(rs_id, chrom, pos, p_value, beta)*

- **Genotype Preprocessing**:
  - Used **bcftools** to filter SNPs based on:
    - `MAF > 0.001`
    - `Missing Rate < 3%`
    - `Hardy-Weinberg Equillibrium (HWE) > 1e-6`
  - SNPs encoded as dosage values (0, 1, 2)

- **Feature Selection**:
  - GWAS p-value filtering (`p < 1e-5`)
  - `SelectKBest` using chi-square or ANOVA F-score

- **Modelling**:
  - **Logistic Regression (Lasso)**
  - Scoring Metrics: AUC, sensitivity, specificity
  - PRS calculated as a weighted sum:
    - PRSᵢ = Σⱼ=1ᵏ (βⱼ × dosageᵢⱼ)

- **Evaluation**:
  - ROC AUC range for SNP-only models *(50/50 split)*:
    - RA: ~0.55
    - ALZ: ~0.60
    - T2D: ~ 0.53
    - HYP: ~0.53
  - ROC AUC range for SNP-only models *(10/90 split)*:
    - RA: ~0.60
    - ALZ: ~0.64
    - T2D: ~ 0.62
    - HYP: ~0.52

## Tools and Libraries

- Python (pandas, scikit-learn, seaborn, matplotlib, numpy)
- bcftools
- Jupyter Notebook

## Linked Resources

- [1000 Genomes Project](https://www.internationalgenome.org/)
- [GWAS Catalog](https://www.ebi.ac.uk/gwas/)
- [bcftools](https://github.com/samtools/bcftools)

## Acknowledgments

- This project was completed under the guidance of **Prof. M.K. Bhuyan**, Department of EEE, **IIT Guwahati**.
- Thanks to [Pragyan Thapa](https://github.com/pragyanthapa) for collaborating on model development and experiments.
