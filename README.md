# Airline Tweet Sentiment Analysis

**ML Internship Take-Home Assignment**  
**Final Performance**: 74.38% weighted F1-score using SVM with RBF kernel

---

## 🚀 Quick Start

<<<<<<< Updated upstream
### View Key Results:
- **📊 Confusion Matrix**: `final_evaluation/confusion_matrix.png`
- **📈 Performance Metrics**: `final_evaluation/class_performance.png` 
- **📝 Reflection**: `reflection.md`
- **📋 Complete Summary**: `ASSIGNMENT_SUMMARY.md`
=======
### 📓 **Primary Deliverable** (Assignment Requirement):
**`main_analysis.ipynb`** - Complete step-by-step analysis notebook
>>>>>>> Stashed changes

### 📋 **Setup Instructions:**

<<<<<<< Updated upstream
=======
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download required data files** (not in git due to size):
   - [Training data](https://drive.google.com/file/d/1iqfE_thVL0JIg77aa5SZTc3OuqctLXLR/view?usp=drive_link) → `data/tweet_sentiment.train.jsonl`
   - [Test data](https://drive.google.com/file/d/1EjSbweOB0ihPHpMKVkcfEqUGd-L1wLwX/view?usp=drive_link) → `data/tweet_sentiment.test.jsonl`
   - [GloVe embeddings](https://drive.google.com/file/d/1t2TXAO-OSrdiQeZPHCz14-2eNViNW4QG/view?usp=drive_link) → `embeddings/glove.6B.100d.txt`

3. **Run the main deliverable:**
   ```bash
   jupyter notebook main_analysis.ipynb
   ```

### 📊 **Additional Resources:**
- **📈 Results & Visualizations**: `docs/` - Pre-generated evaluation metrics and confusion matrices
- **📝 Reflection**: `reflection.md` - Detailed analysis and lessons learned

### 🔬 **Optional: Run Additional Experiments**
```bash
cd experiments
python experiment_runner.py        # Systematic model comparison
python data_augmentation.py        # Bonus: Data augmentation experiment
```

>>>>>>> Stashed changes
---

## 📂 Project Structure

```
├── 📓 main_analysis.ipynb     # 🎯 PRIMARY DELIVERABLE - Complete analysis notebook
├── 📝 reflection.md           # Required reflection section  
├── 📁 src/                    # Core implementation modules
<<<<<<< Updated upstream
├── 📁 experiments/            # Systematic experimentation framework  
├── 📁 final_evaluation/       # Assignment deliverables & visualizations
├── 📁 notebooks/              # EDA and analysis notebooks
├── 📁 data/                   # Tweet datasets  
├── 📁 embeddings/             # GloVe embeddings
├── reflection.md              # ✅ Required reflection section
├── complete_evaluation.py     # Final evaluation script
└── ASSIGNMENT_SUMMARY.md      # Complete assignment overview
=======
├── 📁 docs/                   # Pre-generated evaluation metrics & visualizations
├── 📁 experiments/            # Optional: Additional experiments & data augmentation
├── 📁 data/                   # Tweet datasets (download required)
├── 📁 embeddings/             # GloVe embeddings (download required)
└── requirements.txt           # Dependencies
>>>>>>> Stashed changes
```

---

## ✅ Assignment Requirements Completed

<<<<<<< Updated upstream
| Requirement | Status | Location |
|-------------|--------|----------|
| GloVe-based approach | ✅ Complete | `src/embeddings.py` |
| CPU-friendly models | ✅ Complete | SVM, Logistic Regression, etc. |
| Confusion matrix | ✅ Complete | `final_evaluation/confusion_matrix.png` |
| Misclassification analysis | ✅ Complete | See evaluation output |
| Reflection section | ✅ Complete | `reflection.md` |
| Systematic methodology | ✅ Complete | `experiments/` framework |
=======
- **✅ Runnable Jupyter notebook** → `main_analysis.ipynb` 
- **✅ Concise README with setup** → This file
- **✅ Evaluation metrics & confusion matrix** → Displayed in notebook and `docs/`
- **✅ Optional bonus: Data augmentation** → `experiments/data_augmentation.py`
>>>>>>> Stashed changes

---

## 📊 Key Results

### Systematic Experiment Results:
- **Experiments Conducted**: 10+ different model configurations
- **Performance Range**: 65.29% - 74.38% F1-score
- **Winner**: SVM with RBF kernel (74.38% F1)
- **Runner-up**: Logistic Regression with sum aggregation (73.94% F1)
- **Methodology**: Cross-validation, reproducible seeds, systematic comparison

### Final Model Performance:
- **Model**: SVM with RBF kernel
- **Weighted F1**: **74.38%** (excellent for CPU-friendly models)
- **Accuracy**: 73.57%
- **Training Time**: ~30 seconds on CPU

### Per-Class Performance:
- **Negative**: 81.88% F1 (strongest)
- **Positive**: 64.11% F1  
- **Neutral**: 60.01% F1 (most challenging due to class imbalance)

### Dataset:
- **Training**: 11,712 airline tweets
- **Testing**: 2,928 airline tweets  
- **Classes**: Negative (62.7%), Neutral (21.2%), Positive (16.1%)

---

## 🔬 Technical Highlights

### Systematic Approach:
- 10+ model configurations tested
- Cross-validation for reliable estimates
- Reproducible methodology with fixed seeds

### Engineering Excellence:
- Modular, maintainable architecture
- Comprehensive caching system
- Detailed logging and evaluation

### ML Best Practices:
- Class imbalance handling (`class_weight='balanced'`)
- Feature scaling for SVM optimization
- Comprehensive error analysis

---

## 📈 Performance Context

The **74.38% F1-score** places the model in the **"good performance"** category:

- Basic sentiment models: 60-65% F1
- **Good performance: 70-75% F1** ← This result
- State-of-the-art: 80-85% F1 (transformer models)

<<<<<<< Updated upstream
This represents excellent performance for CPU-friendly models within assignment constraints.

=======
>>>>>>> Stashed changes
