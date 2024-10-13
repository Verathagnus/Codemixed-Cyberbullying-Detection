# Codemixed-Cyberbullying-Detection 

## Introduction

Codemixed Cyberbullying Detection is a machine learning project aimed at identification of the class of cyberbullying in codemixed text samples.

## Formal Description of Task 
Given a text sentence, classify the cyberbullying class present in the test from among 'abusive', 'age', 'gender', 'mockery', 'offensive', 'religion' and 'not_cyberbullying'

## Techniques 
Both Machine Learning and Deep Learning techniques were tested for comparison. Machine Learning models used are Logistic Regression, Naive Bayes Model (Gaussian and Multinomial), Support Vector Machine, Random Forest, XG Boost and Voting Classifier based on the best parameters of the rest of these models. 
The Deep Learning models used are RNN, LSTM, BERT, mBERT and Llama 3.1 8B instruct model.
The detailed parameters and test dataset evaluation summary are described in later sections.

## Tokenizers

The machine learning models and lstm models used BERT tokenizer pretrained on Hinglish data for getting embeddings of length 768 from input sentences. The embeddings were then compressed to 64 length embeddings which were finally used by the models.

The rest of the deep learning models, BERT, mBERT and LLama 3.1 8B Instruct models used their respective pretrained tokenizers.

### Detailed Model Descriptions

## Machine Learning Models
For machine learning models, various parameter ranges for each model were selected and the best parameter options for each model were found using GridSearchCV.

| Model                       | Train Accuracy | Test Accuracy | Precision | Recall  | F1 Score |
|------------------------------|----------------|---------------|-----------|---------|----------|
| **Logistic Regression**       | 0.5532         | 0.5298        | 0.5097    | 0.5112  | 0.5298   |
| **Random Forest**             | 0.9999         | 0.5571        | 0.5895    | 0.5217  | 0.5571   |
| **XGBoost**                   | 0.9999         | 0.5740        | 0.5608    | 0.5615  | 0.5740   |
| **Gaussian Naive Bayes**      | 0.4746         | 0.4626        | 0.4716    | 0.4364  | 0.4626   |
| **Multinomial Naive Bayes**   | 0.3372         | 0.3463        | 0.1141    | 0.2412  | 0.3463   |
| **Support Vector Machine**    | 0.7798         | 0.5877        | 0.5820    | 0.5670  | 0.5877   |
| **Voting Classifier (Hybrid)**| 0.9158         | 0.5713        | 0.5735    | 0.5497  | 0.5713   |

### 1. Logistic Regression

**Hyperparameters:**
- `C`: 0.1
- `multi_class`: 'auto'
- `penalty`: 'l1'
- `solver`: 'saga'

**Performance:**
- **Train Accuracy**: 0.5532
- **Test Accuracy**: 0.5298
- **Precision**: 0.5097
- **Recall**: 0.5112
- **F1 Score**: 0.5298

**Description**: Logistic Regression with L1 regularization and the 'saga' solver. It shows moderate performance on the test set, with balanced precision and recall scores.

---

### 2. Random Forest

**Hyperparameters:**
- `criterion`: 'gini'
- `max_depth`: None
- `max_features`: 'sqrt'
- `min_samples_leaf`: 1
- `min_samples_split`: 2
- `n_estimators`: 4000

**Performance:**
- **Train Accuracy**: 0.9999
- **Test Accuracy**: 0.5571
- **Precision**: 0.5895
- **Recall**: 0.5217
- **F1 Score**: 0.5571

**Description**: Random Forest with 4000 trees, Gini criterion, and square root feature selection. Despite its high accuracy on the training data (overfitting), the test performance shows a more moderate result with a balanced precision and recall.

---

### 3. XGBoost

**Hyperparameters:**
- `colsample_bytree`: 1.0
- `learning_rate`: 0.1
- `max_depth`: 7
- `n_estimators`: 4000
- `reg_alpha`: 0.1
- `reg_lambda`: 0
- `subsample`: 0.6

**Performance:**
- **Train Accuracy**: 0.9999
- **Test Accuracy**: 0.5740
- **Precision**: 0.5608
- **Recall**: 0.5615
- **F1 Score**: 0.5740

**Description**: XGBoost is configured with a learning rate of 0.1, max depth of 7, and 4000 estimators. The model provides good test accuracy and balanced precision-recall. It also exhibits overfitting due to its perfect accuracy on the training set.

---

### 4. Gaussian Naive Bayes

**Hyperparameters:**
- `var_smoothing`: 1e-100

**Performance:**
- **Train Accuracy**: 0.4746
- **Test Accuracy**: 0.4626
- **Precision**: 0.4716
- **Recall**: 0.4364
- **F1 Score**: 0.4626

**Description**: Gaussian Naive Bayes model with a very low variance smoothing parameter (1e-100). The model shows lower performance, but it's still competitive, with slightly better precision than recall.

---

### 5. Multinomial Naive Bayes

**Hyperparameters:**
- `alpha`: 150
- `fit_prior`: True

**Performance:**
- **Train Accuracy**: 0.3372
- **Test Accuracy**: 0.3463
- **Precision**: 0.1141
- **Recall**: 0.2412
- **F1 Score**: 0.3463

**Description**: Multinomial Naive Bayes with an `alpha` of 150. This model performs poorly with a very low precision score, indicating it's not ideal for this particular task.

---

### 6. Support Vector Machine (SVM)

**Hyperparameters:**
- `C`: 1.4
- `class_weight`: None
- `coef0`: 0.0
- `degree`: 2
- `gamma`: 0.01
- `kernel`: 'rbf'
- `probability`: True

**Performance:**
- **Train Accuracy**: 0.7798
- **Test Accuracy**: 0.5877
- **Precision**: 0.5820
- **Recall**: 0.5670
- **F1 Score**: 0.5877

**Description**: SVM with an RBF kernel and a regularization parameter `C` of 1.4. This model shows strong performance on the test set, with well-balanced precision and recall values.

---

### 7. Voting Classifier (Hybrid)

**Ensemble Model:** Combining Logistic Regression, Random Forest, XGBoost, Naive Bayes, and SVM.

**Performance:**
- **Train Accuracy**: 0.9158
- **Test Accuracy**: 0.5713
- **Precision**: 0.5735
- **Recall**: 0.5497
- **F1 Score**: 0.5713

**Description**: The Voting Classifier aggregates the predictions from Logistic Regression, Random Forest, XGBoost, Naive Bayes, and SVM. It provides stable results with a strong trade-off between precision and recall. The high training accuracy suggests some overfitting, but the test results show balanced overall performance.

---

The models implemented in this project vary in performance, with SVM, XGBoost, and the Voting Classifier providing the best results based on test accuracy and balanced F1 scores.

## Deep Learning
For Deep Learning, parameters had to be tuned manually till an appreciable performance was achieved. 


### 1. LSTM Model

**Parameters:**
- Optimizer: Adam
- Learning Rate: 1e-4
- Epochs: 30

**Performance:**
- **Train Accuracy**: 0.9384
- **Test Accuracy**: 0.5379
- **F1 Score**: 0.5379
- **Precision**: 0.5307
- **Recall**: 0.5435

**Description**: The LSTM model was trained using the Adam optimizer and performed well during training but exhibited moderate test accuracy and F1 score, suggesting overfitting. Precision and recall were closely matched, indicating a consistent classification performance across positive cases.

---

### 2. BERT Model

**Parameters**: 
- Optimizer: AdamW (default)
- Learning Rate: 5e-5
- Epochs: Trained until epoch 81, with no improvement observed for the last 10 epochs.

**Performance:**
- **Train Accuracy**: 0.9880
- **Test Accuracy**: 0.7482
- **F1 Score**: 0.7439
- **Precision**: 0.7475
- **Recall**: 0.7482

**Description**: BERT performed excellently during training, reaching near-perfect accuracy. Test accuracy and F1 score were also strong, highlighting BERT’s ability to generalize well. The model showed balanced precision and recall, which means it handled both true positives and false positives well. However, further tuning might improve its stability after long epochs of training.

---

### 3. mBERT Model

**Parameters**:
- Optimizer: AdamW (default)
- Learning Rate: 5e-5 (default parameters similar to BERT)
- Epochs: Trained for 100 epochs

**Performance:**
- **Train Accuracy**: 0.9960
- **Test Accuracy**: 0.7586
- **F1 Score**: 0.7554
- **Precision**: 0.7584
- **Recall**: 0.7586

**Description**: mBERT, the multilingual version of BERT, was trained for 100 epochs and demonstrated outstanding performance. With high test accuracy and balanced precision and recall, mBERT proved to be a robust model for this dataset. The longer training cycle appeared to have solidified its ability to generalize, with only minor differences between train and test performance.

---

### 4. Llama 3.1 8B Instruct Model

**Parameters**:
- Framework: Unsloth with LoRA adapter
- LoRA Parameters: r=64, alpha=64
- Precision: fp16
- Epochs: 30
- Optimizer: AdamW_8bit
- Learning Rate: 2e-4

**Performance**:
- **Train Accuracy**: 0.9373
- **Test Accuracy**: 0.7324
- **F1 Score**: 0.7324
- **Precision**: 0.6652
- **Recall**: 0.6582

**Description**: The Llama 3.1 8B Instruct model was fine-tuned using the Unsloth framework and LoRA adapters, making it highly efficient in terms of parameter utilization. Although the test accuracy and F1 score were slightly lower than BERT and mBERT, it performed admirably, showing consistent precision and recall. With further fine-tuning, the balance between precision and recall could be improved.

---

### Summary Table:

| Model                       | Train Accuracy | Test Accuracy | Precision    | Recall      | F1 Score   |
|------------------------------|----------------|---------------|--------------|-------------|------------|
| **LSTM**                     | 0.9384         | 0.5379        | 0.5307       | 0.5435      | 0.5379     |
| **BERT**                     | 0.9880         | 0.7482        | 0.7475       | 0.7482      | 0.7439     |
| **mBERT**                    | 0.9960         | 0.7586        | 0.7584       | 0.7586      | 0.7554     |
| **Llama 3.1 8B Instruct**     | 0.9373         | 0.7324        | 0.6652       | 0.6582      | 0.7324     |


 

<!-- ## Example

![Streamlit App Screenshot](screenshot.png)  # Add a screenshot of your Streamlit app here -->

## Future Work

- **Extend Language Support**: Increase the number of Indic languages supported by training the models on additional language datasets.
- **Improve Model Performance**: Experiment with advanced models and techniques to enhance the accuracy and robustness of emotion detection.
- **User Feedback**: Incorporate user feedback to continuously improve the application and its usability.

## Contributors

- <h2>Bishwaraj Paul</h2>
  <p><strong>Role: </strong>Intern<br>
  Email: bishwaraj.paul98@gmail.com / bishwaraj.paul@bahash.in<br>
  </p>
- <h2>Dr. Sahinur Rahman Laskar</h2>
  <p><strong>Role:</strong> Mentor<br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
---
