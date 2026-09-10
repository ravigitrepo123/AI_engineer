#  🧠 Artificial Neural Network 

An interactive **Artificial Neural Network (ANN)** web application built using **TensorFlow**, **Keras**, and **Streamlit**.  
This app allows users to upload a dataset, preprocess it automatically, train a neural network model, visualize training progress, and analyze results — all in one interface.

---

## 🚀 Overview

This project demonstrates how to:
- Build and train a fully connected **Deep Learning classification model**.
- Integrate **TensorFlow/Keras** with an interactive **Streamlit** frontend.
- Preprocess, train, and evaluate the model dynamically with user input.
- Visualize **training loss**, **accuracy**, and **classification metrics** in real time.

---

## 🧩 Features

✅ Upload your own CSV dataset  
✅ Automatic data preprocessing (train-test split + scaling)  
✅ Choose from 5 Optimizers (`Adam`, `Adagrad`, `Adamax`, `Adadelta`, `RMSprop`)  
✅ Customize number of epochs via sidebar  
✅ Live training progress visualization  
✅ Early stopping for better generalization  
✅ Model evaluation with classification report and accuracy metrics  

---

## 🏗️ Tech Stack

| Technology | Description |
|-------------|--------------|
| **Python 3.8+** | Core Programming Language |
| **TensorFlow / Keras** | Deep Learning Framework |
| **Streamlit** | Web App Framework |
| **Scikit-learn** | Data Preprocessing & Metrics |
| **Matplotlib** | Visualization |
| **Pandas / NumPy** | Data Handling |

---

## 🧠 Model Architecture
Input Layer: number of features
↓
Dense(16, activation='relu')
↓
Dense(8, activation='relu') + Dropout(0.25)
↓
Dense(4, activation='relu') + Dropout(0.5)
↓
Dense(1, activation='sigmoid')

# yaml
**Loss Function:** Binary Crossentropy  
**Evaluation Metric:** Accuracy  
**Callback Used:** EarlyStopping (patience=20, min_delta=0.001)

---


# Install Dependencies
pip install -r requirements.txt

# Run the Streamlit App
streamlit run app.py


Then open your browser and go to 👉 http://localhost:8501/

# 📊 How to Use

Launch the Streamlit app.

Upload your dataset file (heart_failure_clinical_records_dataset.csv).

Choose optimizer and number of epochs from the sidebar.

Click “Train Model and Analyze Results”.

# View:

Model loss & accuracy plots

Final accuracy score

Classification report table

# 🧾 Example Output
# 📈 Model Training Visualization

Left chart: Loss vs Epochs

Right chart: Accuracy vs Epochs

# 📊 Evaluation Metrics (Example)
Metric	Precision	Recall	F1-Score
Class 0	0.89	0.86	0.88
Class 1	0.82	0.86	0.84
Accuracy			85.20%
📚 Dataset Information

# Dataset Used: Heart Failure Clinical Records Dataset
📦 Source: Kaggle Dataset Link

This dataset contains medical records of patients with heart failure, used to predict the DEATH_EVENT variable (binary classification).


# 💡 Future Improvements

Add multi-class classification support

Include model saving/loading options

Deploy app on Streamlit Cloud / Hugging Face Spaces / Heroku

Add confusion matrix and ROC-AUC plots

<img width="1900" height="927" alt="Screenshot (177)" src="https://github.com/user-attachments/assets/572ce677-d895-4796-aaea-4466faa954bd" />



