Classification of Tweets Using BERT and SVM

Overview

This project focuses on classifying tweets using a combination of BERT (Bidirectional Encoder Representations from Transformers) and SVM (Support Vector Machine). The goal is to leverage BERT's deep contextual understanding of text and SVM's classification capabilities to achieve high accuracy.

Features

Data preprocessing and cleaning of tweets

Fine-tuning BERT for feature extraction

Training an SVM classifier on BERT-generated embeddings

Installation

To set up the project, follow these steps:

Clone the repository:

git clone https://github.com/AlphaDiallo58/Classification_Tweet_With_BERT_n_SVM.git
cd Classification_Tweet_With_BERT_n_SVM

Create a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Usage

Run the main script to preprocess data, extract features using BERT, and classify tweets using SVM:

python main.py

Alternatively, if using Jupyter Notebook, open and execute notebooks/classification_pipeline.ipynb.

You can directly test the trained model svm_model.pkl and improve it based on your own dataset and requirements.

Contributions

Contributions are welcome! If you have improvements or suggestions, feel free to submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

