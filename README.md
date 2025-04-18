# 🧠 Alzheimer's Disease Classification using CNN Features and Machine Learning

This research project develops a hybrid approach for classifying different stages of Alzheimer's Disease from MRI scans. By combining deep learning feature extraction with traditional machine learning classifiers, we achieve high accuracy in identifying disease progression stages.

## 📋 Table of Contents
- [Dataset](#-dataset)
- [Methodology](#️-methodology)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Acknowledgements](#-acknowledgements)

## 📁 Dataset

We used the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset, which consists of MRI scans categorized into five distinct classes representing different stages of dementia:

- **Non-Demented** (Healthy control)
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**
- **Severe Demented**

The dataset is organized into separate directories for each class, with dedicated training and testing splits to ensure robust model evaluation.

## ⚙️ Methodology

Our approach involves a multi-stage pipeline:

### 1. Feature Extraction
- Utilized pre-trained CNN architectures:
  - **VGG16**: Known for excellent feature representation in medical imaging
  - **AlexNet**: Provides complementary feature perspectives
- Features were extracted and combined using the **Hypercolumn technique** to create rich, multi-level representations

### 2. Dimensionality Reduction
Two approaches were implemented and compared:
- **Principal Component Analysis (PCA)**: Reduced feature dimensionality while preserving variance
- **Autoencoders**: Neural network-based non-linear dimensionality reduction

### 3. Classification
Multiple machine learning models were trained and evaluated:
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**
- **Decision Tree (DT)**
- **K-Nearest Neighbors (KNN)**
- **Artificial Neural Network (ANN)**
- **Naive Bayes (NB)**
- **Logistic Regression (LR)**
- **XGBoost**
- **Ensemble Method**

### 4. Evaluation
Performance was assessed using comprehensive metrics:
- **Accuracy**: Overall correctness of classification
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under the Receiver Operating Characteristic curve

## 📊 Results

| Model     | Train Acc | Val Acc | Precision | Recall | F1-Score | AUC    |
|:---------:|:---------:|:-------:|:---------:|:------:|:--------:|:------:|
| RF        | 1.00      | 0.91    | 0.93      | 0.91   | 0.92     | 0.99   |
| SVM       | 0.83      | 0.75    | 0.77      | 0.75   | 0.75     | 0.95   |
| DT        | 1.00      | 0.83    | 0.81      | 0.83   | 0.81     | 0.89   |
| KNN       | 0.83      | 0.70    | 0.71      | 0.70   | 0.70     | 0.93   |
| ANN       | 1.00      | 0.86    | 0.86      | 0.86   | 0.86     | 0.97   |
| NB        | 0.36      | 0.32    | 0.40      | 0.32   | 0.29     | 0.63   |
| LR        | 0.64      | 0.58    | 0.58      | 0.58   | 0.58     | 0.85   |
| XGBoost   | 1.00      | 0.90    | 0.91      | 0.90   | 0.90     | 0.99   |
| Ensemble  | 0.43      | 0.45    | 0.48      | 0.45   | 0.45     | 0.74   |

## 🔍 Key Findings

- **Random Forest** emerged as the best-performing model with a **91% validation accuracy**
- **XGBoost** achieved comparable performance with a **90% validation accuracy**
- Both RF and XGBoost demonstrated excellent ROC-AUC scores of **0.99**
- The **Hypercolumn technique** for feature combination proved highly effective
- **PCA** provided better results than autoencoders for this specific classification task
- **Naive Bayes** showed the poorest performance, suggesting it's not well-suited for this complex classification task

## 📂 Project Structure

```
.
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── feature_extraction.py
│   ├── dimensionality_reduction.py
│   └── classifiers.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── utils/
│   ├── preprocessing.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## 🚀 Installation & Usage

1. Clone the repository
   ```bash
   git clone https://github.com/username/alzheimers-classification.git
   cd alzheimers-classification
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the feature extraction
   ```bash
   python models/feature_extraction.py
   ```

4. Train and evaluate models
   ```bash
   python train.py --model rf --features hypercolumn --reduction pca
   ```

5. Visualize results
   ```bash
   python visualize_results.py
   ```

## 🙏 Acknowledgements

This project was conducted under the guidance of **Mr. Ritesh Jha** as part of our Summer Training program at Birla Institute of Technology, Mesra. We thank the ADNI for providing the dataset used in this research.

---

<p align="center">
  <i>For more information, please contact the project contributors or refer to the associated research paper.</i>
</p>