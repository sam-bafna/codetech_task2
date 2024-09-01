**Name:** Sameksha Bafna  
**Company:** CODTECH IT SOLUTIONS  
**ID:** CT08DS7262  
**Domain:** Machine Learning  
**Duration:** August to September 2024  
**Mentor:** Neela Santhosh Kumar   


# **Project Report: Developing a Sentiment Analysis Model to Classify Movie Reviews**
![image](https://github.com/user-attachments/assets/437a4d00-fd94-41ae-accd-a33e0d2d098d)
![image](https://github.com/user-attachments/assets/7fddd629-2505-4ed4-b0f1-f38c5c642561)


**Title:** *Developing a Sentiment Analysis Model to Classify Movie Reviews as Positive or Negative*  
**Dataset:** *IMDb Movie Reviews*  
**Objective:** *To build and train a machine learning model that can classify movie reviews as positive or negative based on the textual content.*


## **Objective:**

The objective of this project is to develop a sentiment analysis model that can classify movie reviews as either positive or negative. The IMDb Movie Reviews dataset, which contains 50,000 reviews labeled as either positive or negative, was used to train and test the model. The process includes data preprocessing, model construction, training, evaluation, and prediction on new reviews.


## **Key Activities:**

### 1. **Data Collection:**

- The IMDb Movie Reviews dataset was obtained using the `tensorflow_datasets` library, which provided both training and testing data.
- The dataset consists of 25,000 training reviews and 25,000 testing reviews, each labeled with a sentiment (positive or negative).

### 2. **Data Preprocessing:**

- **Tokenization:** The text data was tokenized using the `Tokenizer` class from TensorFlow's Keras API. A vocabulary size of 10,000 was chosen, and an out-of-vocabulary (OOV) token was added to handle words not in the training data.
  
- **Padding:** The tokenized sequences were padded to ensure uniform input lengths. The maximum sequence length was set to 120 words, with padding applied post-sequence.

- **Conversion to Sequences:** The reviews were converted into sequences of integers representing words, and these sequences were padded to a uniform length.

### 3. **Model Building:**

- **Architecture:**
  - **Embedding Layer:** Converts the input sequence of words into dense vectors of fixed size.
  - **Global Average Pooling 1D Layer:** Reduces the dimensionality of the vectors by taking the average over all time steps.
  - **Dense Layer:** A fully connected layer with 16 units and ReLU activation function.
  - **Output Layer:** A single neuron with sigmoid activation for binary classification (positive or negative sentiment).

- **Compilation:** The model was compiled using the Adam optimizer and binary cross-entropy as the loss function, with accuracy as the evaluation metric.

### 4. **Model Training:**

- The model was trained for 10 epochs on the training data, with validation on the testing data to monitor performance and prevent overfitting.

### 5. **Model Evaluation:**

- After training, the model was evaluated on the test set. The evaluation metrics used were loss and accuracy, with the final test accuracy being reported.

### 6. **Prediction Example:**

- The model was used to predict the sentiment of new, unseen movie reviews. Reviews were tokenized, padded, and passed through the trained model to determine if they were positive or negative.


## **Technology Used:**

- **Programming Language:** *Python*
- **Libraries:** 
  - **Data Handling:** *TensorFlow, NumPy*
  - **Modeling and Evaluation:** *TensorFlow Keras (including layers, models, Tokenizer, pad_sequences)*


## **Dataset:**

- **Name:** *IMDb Movie Reviews Dataset*
- **Source:** *Loaded from TensorFlow Datasets (`tensorflow_datasets`)*
- **Key Features Used:**
  - *Textual movie reviews* (input)
  - *Binary sentiment labels* (positive or negative)


## **Conclusion:**

This report outlines the steps taken to develop a sentiment analysis model for classifying movie reviews as positive or negative.
The process included data preprocessing, model construction, training, and evaluation. The model demonstrated good performance on the IMDb dataset,
and was able to predict the sentiment of new reviews accurately.
