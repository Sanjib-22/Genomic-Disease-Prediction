# Genomic Disease Prediction using Polygenic Risk Scores (PRS)

The project mainly focuses on evaluating machine learning models and predicting specific genetically inherited diseases using **Polygenic Risk Scores (PRS)**, leveraging Genomic data and machine learning techniques.
The work was carried out as a part of an internship at **IIT Guwahati** .

## 🔬 Project Overview

The goal of this project is to assess the predicitive capability of **common SNPs (Single Nucleotide Polymorphism)** using PRS, applied across multiple diseases:
- **Rheumatoid Arthritis**
- **Alzheimer's Disease**
- **Type 2 Diabetes**
- **Hyperthyroidism**

## 📁 Project Structure

```
Internship_Project/
├─ Final Datasets/
│  ├─ ALZ_final/
│  │  ├─ ALZ_aligned_genotypes_final.csv
│  │  ├─ ALZ_aligned_phenotypes_final_extended.csv
│  │  └─ ALZ_aligned_phenotypes_final.csv
│  ├─ HYP_final/
│  │  ├─ HYP_aligned_genotypes_final.csv
│  │  ├─ HYP_aligned_phenotypes_final_extended.csv
│  │  └─ HYP_aligned_phenotypes_final.csv
│  ├─ RA_final/
│  │  ├─ RA_aligned_genotypes_final.csv
│  │  ├─ RA_aligned_phenotypes_final_extended.csv
│  │  └─ RA_aligned_phenotypes_final.csv
│  └─ T2D_final/
│     ├─ T2D_aligned_genotypes_final.csv
│     ├─ T2D_aligned_phenotypes_final_extended.csv
│     └─ T2D_aligned_phenotypes_final.csv
├─ Models/
│  ├─ Alzheimers.ipynb
│  ├─ Hyperthyroidism.ipynb
│  ├─ Rheumatoid_Arhtiritis.ipynb
│  └─ Type_2_diabetes.ipynb
├─ Outputs/
│  ├─ ALZ_results/
│  │  ├─ ALZ_confusion_matrix.png
│  │  ├─ ALZ_PRS_distribution_fixed.png
│  │  ├─ ALZ_prs_scores.csv
│  │  ├─ ALZ_selectKbest_logreg_model_AgeSex.joblib
│  │  ├─ ALZ_selectKbest_logreg_model.joblib
│  │  ├─ ALZ_selectKbest_model_performance_AgeSex.png
│  │  ├─ ALZ_selectKbest_model_performance.png
│  │  ├─ logreg_selectKbest_coefficients_AgeSex.csv
│  │  └─ logreg_selectKbest_coefficients.csv
│  ├─ HYP_results/
│  │  ├─ HYP_confusion_matrix.png
│  │  ├─ HYP_PRS_distribution_fixed.png
│  │  ├─ HYP_prs_scores.csv
│  │  ├─ HYP_selectKbest_logreg_model_AgeSex.joblib
│  │  ├─ HYP_selectKbest_logreg_model.joblib
│  │  ├─ HYP_selectKbest_model_performance_AgeSex.png
│  │  ├─ HYP_selectKbest_model_performance.png
│  │  ├─ logreg_selectKbest_coefficients_AgeSex.csv
│  │  └─ logreg_selectKbest_coefficients.csv
│  ├─ RA_results/
│  │  ├─ logreg_selectKbest_coefficients_AgeSex.csv
│  │  ├─ logreg_selectKbest_coefficients.csv
│  │  ├─ RA_confusion_matrix.png
│  │  ├─ RA_PRS_distribution_fixed.png
│  │  ├─ RA_prs_scores.csv
│  │  ├─ RA_selectKbest_logreg_model_AgeSex.joblib
│  │  ├─ RA_selectKbest_logreg_model.joblib
│  │  ├─ RA_selectKbest_model_performance_ageSex.png
│  │  └─ RA_selectKbest_model_performance.png
│  ├─ T2D_results/
│  │  ├─ logreg_selectKbest_coefficients_AgeSex.csv
│  │  ├─ logreg_selectKbest_coefficients.csv
│  │  ├─ T2D_confusion_matrix.png
│  │  ├─ T2D_PRS_distribution_fixed.png
│  │  ├─ T2D_prs_scores.csv
│  │  ├─ T2D_selectKbest_logreg_model_AgeSex.joblib
│  │  ├─ T2D_selectKbest_logreg_model.joblib
│  │  ├─ T2D_selectKbest_model_performance_AgeSex.png
│  │  └─ T2D_selectKbest_model_performance.png
│  └─ Table.ipynb
├─ Raw_Datasets/
│  ├─ ALZ_data/
│  │  ├─ ALZ_chr_dosage_matrix.csv
│  │  ├─ ALZ_chr_genotypes_raw.txt
│  │  ├─ ALZ_chr_genotypes_with_header.tsv
│  │  ├─ ALZ_top2000_cleaned.csv
│  │  ├─ phenotype_ALZ_aligned.csv
│  │  ├─ phenotype_ALZ_simulated_50_50.tsv
│  │  └─ sample_ids(1).txt
│  ├─ HYP_data/
│  │  ├─ HYP_chr_dosage_matrix.csv
│  │  ├─ HYP_chr_genotypes_raw.txt
│  │  ├─ HYP_chr_genotypes_with_header.tsv
│  │  ├─ HyperTH_top2000_cleaned.csv
│  │  ├─ phenotype_HYP_aligned.csv
│  │  ├─ phenotype_HYP_simulated_50_50.tsv
│  │  └─ sample_ids(3).txt
│  ├─ RA_data/
│  │  ├─ phenotype_RA_aligned.csv
│  │  ├─ phenotype_RA_simulated_50_50.tsv
│  │  ├─ RA_aligned_genotypes_final.csv
│  │  ├─ RA_aligned_phenotypes_final.csv
│  │  ├─ RA_chr1_6_dosage_matrix.csv
│  │  ├─ RA_chr1_6_genotypes_raw.txt
│  │  ├─ RA_chr1_6_genotypes_with_header.tsv
│  │  ├─ RA_top10000_cleaned.csv
│  │  └─ sample_ids.txt
│  ├─ T2D_data/
│  │  ├─ phenotype_T2D_aligned.csv
│  │  ├─ phenotype_T2D_simulated_50_50.tsv
│  │  ├─ sample_ids(2).txt
│  │  ├─ T2D_chr_dosage_matrix.csv
│  │  ├─ T2D_chr_genotypes_raw.txt
│  │  ├─ T2D_chr_genotypes_with_header.tsv
│  │  └─ T2D_top2000_cleaned.csv
│  ├─ Dataset_processing_ALZ.ipynb
│  ├─ Dataset_processing_HYP.ipynb
│  ├─ Dataset_processing_RA.ipynb
│  └─ Dataset_processing_T2D.ipynb
├─ .gitignore
├─ environment.yaml
├─ LICENSE
├─ output.md
├─ output.txt
└─ README.md
```

## 🧬 Datasets Used

- **1000 Genomes Project** (genotype data)
- Phenotype labels were **simulated** to follow disease-specific prevalance are used for binary classificaton.
- Two splits were tested:
  - **50/50 Case-Control**
  - **10/90 Case-Control** *(produced more realistic and better-performing results)*

## 🧠 Methodology

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

## ⚒️ Tools and Libraries

- Python (pandas, scikit-learn, matplotlib, numpy)
- bcftools
- Jupyter Notebook

## 🔗 Linked Resources

- [1000 Genomes Project](https://www.internationalgenome.org/)
- [GWAS Catalog](https://www.ebi.ac.uk/gwas/)
- [bcftools](https://github.com/samtools/bcftools)

## 🙌 Acknowledgments

- This project was completed under the guidance of **Prof. M.K. Bhuyan**, Department of EEE, **IIT Guwahati**.
- Thanks to [Pragyan Thapa](https://github.com/pragyanthapa) for collaborating on model development and experiments.
