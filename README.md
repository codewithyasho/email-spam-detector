# SMS/Email Spam Detector

A machine learning-powered web application that detects whether an SMS or email message is spam or legitimate (ham) using natural language processing and machine learning techniques.

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for spam detection that includes:
- Data preprocessing and exploratory data analysis
- Text preprocessing with NLTK
- Machine learning model training
- Interactive web application using Streamlit

## 🚀 Live Demo

Run the Streamlit app locally to test the spam detector:
```bash
streamlit run app.py
```

## 📋 Features

- **Real-time Spam Detection**: Input any text message and get instant spam/ham classification
- **Text Preprocessing**: Comprehensive text cleaning including:
  - Lowercasing
  - Tokenization
  - Special character removal
  - Stopword removal
  - Stemming using Porter Stemmer
- **Machine Learning Model**: Trained classifier with TF-IDF vectorization
- **User-friendly Interface**: Clean and intuitive Streamlit web interface

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **NLTK** - Natural language processing
- **Scikit-learn** - Machine learning library
- **Joblib** - Model serialization

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/codewithyasho/email-spam-detector.git
   cd sms-spam-detector
   ```

2. **Install required packages**:
   ```bash
   pip install streamlit pandas numpy nltk scikit-learn joblib
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```


## 🧠 Model Details

### Text Preprocessing Pipeline:
1. **Lowercasing**: Convert all text to lowercase
2. **Tokenization**: Split text into individual words
3. **Special Character Removal**: Remove emojis and special characters
4. **Stopword Removal**: Remove common English stopwords
5. **Stemming**: Reduce words to their root form using Porter Stemmer

### Machine Learning Pipeline:
1. **TF-IDF Vectorization**: Convert text to numerical features
2. **Model Training**: Trained classifier (specific algorithm in notebook)
3. **Model Evaluation**: Performance metrics and validation

## 🚀 Usage

### Running the Web Application:
```bash
streamlit run app.py
```

### Using the Application:
1. Open the web interface in your browser
2. Enter any SMS or email text in the text area
3. Click "Predict" to get the classification result
4. View whether the message is classified as SPAM or NOT SPAM

### Example Usage in Code:
```python
from streamlit_app import text_preprocessing
import joblib

# Load models
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess and predict
text = "Congratulations! You've won $1000. Click here to claim now!"
preprocessed = text_preprocessing(text)
vectorized = vectorizer.transform([preprocessed])
prediction = model.predict(vectorized)[0]

print("Spam" if prediction == 1 else "Not Spam")
```


## 🔍 Development Process

1. **Data Exploration**: Analysis of the spam dataset characteristics
2. **Data Cleaning**: Handling missing values, duplicates, and data preprocessing
3. **Feature Engineering**: Text preprocessing and TF-IDF vectorization
4. **Model Training**: Training and evaluating multiple algorithms
5. **Model Selection**: Choosing the best performing model
6. **Deployment**: Creating the Streamlit web application

## 📝 Files Description

- **`main.ipynb`**: Complete machine learning pipeline with data analysis, preprocessing, model training, and evaluation
- **`app.py`**: Web application for real-time spam detection
- **`spam_detector_model.pkl`**: Serialized trained machine learning model
- **`tfidf_vectorizer.pkl`**: Serialized TF-IDF vectorizer
- **`spam.csv`**: SMS spam collection dataset
- **`nltk.txt`**: NLTK data requirements (stopwords, punkt)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


