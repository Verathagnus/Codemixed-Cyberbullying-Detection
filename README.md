# Codemixed-Cyberbullying-Detection 

## Introduction

Codemixed Cyberbullying Detection is a machine learning project aimed at identification of the class of cyberbullying in codemixed text samples.

## Formal Description of Task 
Given a text sentence, classify the cyberbullying class present in the test from among 'abusive', 'age', 'gender', 'mockery', 'offensive', 'religion' and 'not_cyberbullying'

## Techniques 
Both Machine Learning and Deep Learning techniques were tested for comparison. Machine Learning models used are Naive Bayes Model (Gaussian and Multinomial), Support Vector Machine, Random Forest, XG Boost and Voting Classifier based on all these models. 
The Deep Learning models used are RNN, LSTM, BERT, mBERT and Llama 3.1 8B instruct model.
The detailed parameters and test dataset evaluation summary are described in later sections.

## Tokenizers

The machine learning models used BERT tokenizer pretrained on Hinglish data for getting embeddings of length 768 from input sentences. The embeddings were then compressed to 64 length embeddings which were finally used by the model.
The deep learning models used a different approach to tokenization.

### Detailed Model Descriptions

For machine learning models, various parameter ranges for each model were selected and the best parameter options for each model were found using GridSearchCV.

## 

For Deep Learning, parameters had to be tuned manually till an appreciable performance was achieved.


## Model Performance

The following table summarizes the performance of each model on both training and test data:

| Model               | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---------------------|----------------|---------------|-----------|--------|----------|
| Naive Bayes         | 0.2597         | 0.25          | 0.35      | 0.25   | 0.27     |
| Logistic Regression | 0.4424         | 0.42          | 0.36      | 0.42   | 0.35     |
| KNN                 | 0.4800         | 0.35          | 0.31      | 0.35   | 0.31     |
| K-Means             | 0.1671         | 0.17          | 0.21      | 0.17   | 0.18     |
| SVM                 | 0.7941         | 0.46          | 0.44      | 0.46   | 0.43     |
| Decision Tree       | 0.5398         | 0.34          | 0.27      | 0.34   | 0.29     |
| Random Forest       | 0.9971         | 0.42          | 0.35      | 0.42   | 0.33     |


## Setup

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/<yourusername>/Multilingual-Emotion-Detection.git
   cd Multilingual-Emotion-Detection
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Download model files at [https://uploadnow.io/f/dc4f6Hb](https://uploadnow.io/f/dc4f6Hb) and place them in a folder named models inside the project directory.**
5. **Run the Streamlit application:**
   ```sh
   streamlit run streamlit_app.py
   ```

## Usage

Once the Streamlit app is running, you can interact with the models through a web interface. You can input text in Hindi (or other supported Indic languages) and select the model you wish to use for emotion detection. The app will display the detected emotion. Using fasttext-langdetect and iso639 packages, the input language is found and then appropriate models list are selected for that language.  

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
