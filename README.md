# Student Dropout Prediction using Sequential Data (OULAD) 

Predicting student dropout before a course ends, by treating Virtual Learning Environment clickstream activity as a time series rather than a static snapshot. Built on Databricks, comparing an LSTM against a Transformer on the same task.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4169E1?style=flat-square)

## Objective

Traditional models often treat student data as a static snapshot and miss the temporal pattern in how engagement builds or fades over a course. This project models each student's VLE click activity as an ordered sequence and predicts final outcome (Pass or Dropout) early enough for a real intervention to matter, using the Open University Learning Analytics Dataset (OULAD).

## Dataset

OULAD: 7 modules, over 32,000 students. The core file is `studentVle.csv`, which logs every click a student makes on the Virtual Learning Environment, joined with `studentInfo.csv` for demographics and the target label. Final result "Fail" or "Withdrawn" is mapped to dropout (1), "Pass" or "Distinction" to non dropout (0).

## Workflow

**1. Data loading with Apache Spark**
Loaded `studentVle`, `studentInfo`, `assessments`, and related files as Spark DataFrames from Unity Catalog volumes, handling over 10 million clickstream entries efficiently at that scale.

**2. Exploratory data analysis**
Confirmed class imbalance in the target (only a small share of students are dropouts), which shaped the choice of evaluation metric toward F1 Score and Recall rather than accuracy. VLE interaction patterns also showed high variance student to student.

**3. Sequence construction**
Converted raw click logs into per student time series using Spark Window functions, ordering clicks by date and aggregating them into a single ordered sequence per student, turning unstructured logs into structured sequence data.

**4. Normalization and PyTorch Dataset class**
Applied Min Max normalization to high range values like `id_site` and `sum_click`. Built a custom PyTorch Dataset class that pads or truncates every sequence to a fixed 200 time steps so batches can be processed uniformly.

**5. LSTM model**
A multilayer LSTM as the baseline sequence model, trained with the Adam optimizer. Applied class weighting in the loss function to penalize misclassifying the minority dropout class.

**6. Transformer model**
A Transformer with multihead self attention, processing the full sequence at once rather than step by step. Required careful gradient clipping and learning rate scheduling to avoid training collapse.

**7. Experiment tracking with MLflow**
Logged hyperparameters, training and validation loss curves, and final metrics for both architectures. Registered the best performing models in the Unity Catalog model registry.

## Results

| Metric | LSTM | Transformer |
|---|---|---|
| Accuracy | 73.50% | 72.90% |
| Precision | 78.59% | 78.03% |
| Recall | 61.22% | 61.22% |
| F1 Score | 68.83% | 68.61% |

**LSTM confusion matrix**

| | Predicted Dropout | Predicted Pass |
|---|---|---|
| **Actual Dropout** | 870 (TP) | 551 (FN) |
| **Actual Pass** | 237 (FP) | 1316 (TN) |

**Transformer confusion matrix**

| | Predicted Dropout | Predicted Pass |
|---|---|---|
| **Actual Dropout** | 881 (TP) | 558 (FN) |
| **Actual Pass** | 248 (FP) | 1287 (TN) |

## Key Insights

Both architectures landed in almost the same place, F1 Score 68.83% for LSTM versus 68.61% for Transformer, and identical Recall (61.22%) on both. Neither model has a clear accuracy edge over the other on this dataset.

**LSTM was chosen as the production model**, for two practical reasons rather than a raw performance win: it edges out the Transformer slightly on F1 Score, and it trains noticeably faster and cheaper than the attention heavy Transformer. At 32,000 students, the extra representational capacity of a Transformer does not pay for its added compute cost. This is the kind of tradeoff worth stating directly rather than defaulting to whichever architecture sounds more advanced.

## Tech Stack

| Layer | Tools |
|---|---|
| Platform | Databricks Runtime for ML, Unity Catalog |
| Big data processing | Apache Spark / PySpark |
| Deep learning | PyTorch |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Visualization | Matplotlib, Seaborn |

## Reproducing This Project

1. Load the OULAD CSVs into a Unity Catalog Volume.
2. Import `notebooks/student_dropout_oulad.ipynb` into your Databricks workspace.
3. Attach a cluster with GPU support if available. The Transformer especially benefits from it, LSTM will still train reasonably on CPU.
4. Run all cells. Sequence construction is the slowest step since it runs a Spark Window function over 10 million rows, everything after that is fast.
5. Compare runs in the MLflow experiment view before deciding which model to register.
