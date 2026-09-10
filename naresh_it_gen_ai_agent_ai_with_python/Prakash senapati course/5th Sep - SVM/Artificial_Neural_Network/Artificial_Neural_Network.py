import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta, RMSprop 
import tensorflow as tf 

st.set_page_config(layout="wide", page_title="Heart Failure ANN Optimizer Comparison")

@st.cache_data
def preprocess_data(df):
    """Splits and scales the data."""
    
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    input_dim = X.shape[1]

    # Split the data 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Standardize numerical features (Crucial for ANNs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames for consistency 
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, input_dim

# ---  Model Definition Function ---
def build_model(input_dim):
    """Defines and returns the Sequential ANN model."""
    model = Sequential()
    
    # First Hidden Layer (Input Layer)
    model.add(Dense(units = 32, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = input_dim))
    
    # Second Hidden Layer
    model.add(Dense(units = 16, kernel_initializer = 'he_uniform', activation = 'relu'))
    model.add(Dense(units = 8, kernel_initializer = 'he_uniform', activation = 'relu'))
    model.add(Dropout(0.15))

    # Third Hidden Layer
    model.add(Dense(units = 4, kernel_initializer = 'he_uniform', activation = 'relu'))
    model.add(Dropout(0.3))

    # Output Layer for Binary Classification (DEATH_EVENT)
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    return model

# --- 3. Sidebar Controls (Inputs) ---
st.sidebar.header("ANN Model Configuration")

# Optimizer selection
optimizer_name = st.sidebar.selectbox(
    "Select Optimizer:",
    ('Adam', 'Adagrad', 'Adamax', 'Adadelta', 'RMSprop')
)

# Learning Rate selection
learning_rate = st.sidebar.number_input(
    "Select Learning Rate (Only affects Adam, Adagrad, Adamax, RMSprop):",
    min_value=0.00001,
    max_value=0.1,
    value=0.001,
    step=0.0001,
    format="%f"
)

# Epochs selection
epochs = st.sidebar.slider("Select Max Number of Epochs:", 50, 500, 200, step=50)

# Batch size is fixed
batch_size = 32

# Early Stopping setup
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
    monitor='val_loss' 
)

if st.sidebar.button("Reset Training State"):
    if 'training_running' in st.session_state:
        st.session_state['training_running'] = False
        st.sidebar.success("Training state reset. You can try training again.")
    else:
        st.sidebar.info("Training state was already clear.")

# ---  Main App Content and File Uploader ---

st.title("Artificial Neural Network")

uploaded_file = st.file_uploader(
    "Upload the 'heart_failure_clinical_records_dataset.csv' file", 
    type="csv"
)

# Initialize variables to None
X_train, X_test, y_train, y_test, input_dim = None, None, None, None, 0

# Process the file only if it is uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'DEATH_EVENT' not in df.columns:
             st.error("The uploaded CSV must contain the column 'DEATH_EVENT' for classification.")
        else:
            X_train, X_test, y_train, y_test, input_dim = preprocess_data(df)
            st.success("Data loaded and preprocessed successfully. Ready for training.")

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        uploaded_file = None 

# --- Training Execution ---

if uploaded_file is None:
    st.info("Please upload the CSV file using the widget above to proceed.")
else:
    # Display model info when data is ready
    st.markdown("### Model Structure Overview")
    st.markdown(f"""
    * **Input Dimension:** {input_dim} features
    * **Architecture:** **32** $\\rightarrow$ **16** $\\rightarrow$ **8** (Dropout 0.15) $\\rightarrow$ **4** (Dropout 0.3) $\\rightarrow$ 1 (Sigmoid)
    * **Initialization:** All ReLU layers now use `'he_uniform'`
    * **Loss:** Binary Crossentropy
    * **Callbacks:** Early Stopping (patience=20, min\_delta=0.001, restores best weights)
    """)
    
    st.markdown("---")
    st.markdown("Use the sidebar to configure the optimizer and training epochs, then click 'Train Model' to run the experiment and view the results.")


if st.sidebar.button("Train Model and Analyze Results"):
    if X_train is None or input_dim == 0:
        st.error("Please upload the data file first before attempting to train the model.")
    else:
        # State management check
        if 'training_running' not in st.session_state:
            st.session_state['training_running'] = False

        if not st.session_state['training_running']:
            st.session_state['training_running'] = True
            
            # --- Training Block ---
            try:
                with st.spinner(f"Training model with {optimizer_name} for {epochs} epochs..."):

                    tf.keras.backend.clear_session()

                    #  optimizer with selected learning rate
                    if optimizer_name == 'Adam':
                        optimizer = Adam(learning_rate=learning_rate)
                    elif optimizer_name == 'Adagrad':
                        optimizer = Adagrad(learning_rate=learning_rate)
                    elif optimizer_name == 'Adamax':
                        optimizer = Adamax(learning_rate=learning_rate)
                    elif optimizer_name == 'Adadelta':
                        optimizer = Adadelta(learning_rate=1.0) 
                    elif optimizer_name == 'RMSprop':
                        optimizer = RMSprop(learning_rate=learning_rate)
                    
                    #  Build  the Model
                    model = build_model(input_dim)
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                    
                    #  Train the Model
                    start_time = time.time()
                    
                    history = model.fit(
                        X_train.values, y_train.values, 
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping],
                        validation_split=0.2, 
                        verbose=0 
                    )
                    
                    end_time = time.time()
                    training_time = end_time - start_time
                
                st.success(f"Training Complete! Time taken: {training_time:.2f} seconds.")

                # --- Visualization (Training History) ---
                st.header("Training History Visualization")
                col1, col2 = st.columns(2)

                # Loss Plot
                with col1:
                    st.subheader(f"Loss Progression ({optimizer_name})")
                    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
                    ax_loss.plot(history.history['loss'], label='Training Loss', color='#FF6347')
                    ax_loss.plot(history.history['val_loss'], label='Validation Loss', color='#4682B4')
                    ax_loss.set_title('Model Loss vs. Epochs')
                    ax_loss.set_xlabel('Epoch')
                    ax_loss.set_ylabel('Loss (Binary Crossentropy)')
                    ax_loss.legend()
                    ax_loss.grid(True)
                    st.pyplot(fig_loss)

                # Accuracy Plot
                with col2:
                    st.subheader(f"Accuracy Progression ({optimizer_name})")
                    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
                    ax_acc.plot(history.history['accuracy'], label='Training Accuracy', color='#3CB371')
                    ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#8A2BE2')
                    ax_acc.set_title('Model Accuracy vs. Epochs')
                    ax_acc.set_xlabel('Epoch')
                    ax_acc.set_ylabel('Accuracy')
                    ax_acc.legend()
                    ax_acc.grid(True)
                    st.pyplot(fig_acc)

                # --- Model Evaluation ---
                st.header("Model Evaluation on Test Set")

                # Predict on the test set
                y_pred_proba = model.predict(X_test.values, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype("int32")

                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                
                col3, col4 = st.columns([1, 2])

                with col3:
                    st.markdown(f"**Final Test Set Accuracy:**")
                    st.metric(label="Accuracy Score", value=f"{test_accuracy*100:.2f} %")
                    st.markdown(f"**Epochs Executed:** {len(history.history['loss'])} / {epochs}")
                    st.markdown(f"**Optimizer:** {optimizer_name}")
                    st.markdown(f"**Learning Rate Used:** `{learning_rate}`") 

                with col4:
                    st.markdown("**Classification Report**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    st.dataframe(report_df)
                
            except Exception as e:
                st.error(f"An error occurred during training or evaluation: {e}")
                
            finally:
                try:
                    if 'model' in locals():
                        del model 
                    tf.keras.backend.clear_session()
                except Exception:
                    pass 
                
                st.session_state['training_running'] = False
        else:
            st.info("Training is already running. Please wait for the current run to complete.")


