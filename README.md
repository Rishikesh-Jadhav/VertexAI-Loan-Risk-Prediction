# Vertex AI: Predicting Loan Risk

 This project demonstrates the practical application of **Google Cloud's Vertex AI** to build, train, and deploy a machine learning model aimed at predicting loan repayment risks using a tabular dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Project Steps](#project-steps)
  - [Task 1: Prepare the Training Data](#task-1-prepare-the-training-data)
    - [1.1 Create a Dataset](#11-create-a-dataset)
    - [1.2 Upload Data](#12-upload-data)
    - [1.3 Generate Statistics (Optional)](#13-generate-statistics-optional)
  - [Task 2: Train Your Model](#task-2-train-your-model)
    - [2.1 Initiate Model Training](#21-initiate-model-training)
    - [2.2 Configure Training Settings](#22-configure-training-settings)
    - [2.3 Define Model Details](#23-define-model-details)
    - [2.4 Select Features for Training](#24-select-features-for-training)
    - [2.5 Configure Compute and Pricing](#25-configure-compute-and-pricing)
  - [Task 3: Evaluate the Model Performance](#task-3-evaluate-the-model-performance-demonstration-only)
  - [Task 4: Deploy the Model](#task-4-deploy-the-model-demonstration-only)
  - [Task 5: SML Bearer Token](#task-5-sml-bearer-token)
  - [Task 6: Get Predictions](#task-6-get-predictions)
- [Key Terminologies](#key-terminologies)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

This project uses **Vertex AI**, Google Cloud's unified machine learning platform, to predict loan repayment risks. Vertex AI simplifies the process of training machine learning models by automating feature engineering, model selection, and hyperparameter tuning. The primary goal is to build a classification model that accurately predicts whether a loan applicant will repay their loan or default based on historical data.

---

## Objectives

1. **Upload a Dataset to Vertex AI:** Prepare and import tabular data into Vertex AI for model training.
2. **Train a Machine Learning Model with AutoML:** Utilize Vertex AI's capabilities to build a classification model.
3. **Evaluate Model Performance:** Understand key evaluation metrics to assess model effectiveness.
4. **Deploy the Model:** (Demonstration Only) Learn the steps to deploy a trained model to an endpoint for serving predictions.
5. **Authenticate and Make Predictions:** Use a Bearer Token to securely interact with the deployed model and obtain predictions.

---

## Project Steps

### Task 1: Prepare the Training Data

#### 1.1 Creating a Dataset

**Objective:** Created a dataset in Vertex AI named **LoanRisk** to store and manage training data.

#### 1.2 Upload Data

**Objective:** Import the loan risk dataset into Vertex AI from Google Cloud Storage.
Generated statistics to obtain descriptive statistics for each column. This helps in understanding data distribution and identifying anomalies.
   - Each column showed detailed analytical charts.

#### 1.3 Generate Statistics (Optional)

**Objective:** Obtain descriptive statistics for each column to better understand the dataset.

---

### Task 2: Train Your Model

**Objective:** Trained a classification model to predict whether a customer will repay a loan using Vertex AI's AutoML.

#### 2.1 Initiate Model Training

#### 2.2 Configure Training Settings

**Objective:** "Classification"** for predicting a categorical outcome (`0` for repay, `1` for default).

#### 2.3 Define Model Details

**Explored Advanced Options**
   - Configured how to split your data into training and testing sets.
   - Specified encryption settings.

#### 2.4 Select Features for Training

1. **Add Features:**
   - Customized which columns to include.

2. **Exclude Irrelevant Features:**
   - For instance, **ClientID** was irrelevant for predicting loan risk hence was excluded while the training model.

3. **Explored Advanced Options:**
   - Explored additional optimization objectives and feature engineering options as needed.

#### 2.5 Configure Compute and Pricing

1. **Set Budget:**
   - **Budget:** Entered `1` to allocate **1 node hour** for training.
2. **Enabled Early Stopping:**
   - **Early Stopping:**  To allow the training process to halt early if it converges, saving compute resources.

---

### Task 3: Evaluate the Model Performance 

**Objective:** Understand how to evaluate model performance using Vertex AI's evaluation metrics.

#### Key Evaluation Metrics:

1. **Precision/Recall Curve:**
   - **Precision:** Measures the accuracy of positive predictions.
   - **Recall:** Measures the ability to find all positive instances.
   - **Trade-off:** Adjusting the **confidence threshold** affects precision and recall. A higher threshold increases precision but decreases recall, and vice versa.

2. **Confusion Matrix:**
   - **True Positives (TP):** Correctly predicted positives.
   - **True Negatives (TN):** Correctly predicted negatives.
   - **False Positives (FP):** Incorrectly predicted positives.
   - **False Negatives (FN):** Incorrectly predicted negatives.
   - **Usage:** Helps visualize the performance of the classification model.

3. **Feature Importance:**
   - **Description:** Displays how much each feature contributes to the model's predictions.
   - **Visualization:** Typically shown as a bar chart where longer bars indicate higher importance.
   - **Application:** Useful for feature selection and understanding model behavior.

---

### Task 4: Deploying the Model 

**Objective:** Understand the steps required to deploy your trained model to an endpoint for serving predictions.

#### Steps to Deploy the Model:

1. **Initiate Deployment:**
 
2. **Configure Endpoint Details:**
   - **Endpoint Name:** `LoanRisk`.
   - **Model Settings:**
     - **Traffic Splitting:** Leave as default unless you have specific requirements.
     - **Machine Type:** Choose `e2-standard-8` (8 vCPUs, 32 GiB memory) for robust performance.
     - **Explainability Options:** Enable **Feature attribution** to understand feature contributions.

3. **Ready for Predictions:**
   - Once deployed, model was ready to serve predictions through the endpoint.

---

### Task 5: SML Bearer Token

**Objective:** Obtain a **Bearer Token** to authenticate and authorize requests to the deployed model endpoint.

---

### Task 6: Get Predictions

**Objective:** Use the **Shared Machine Learning (SML)** service to make predictions with your trained model.

#### Step 6.1: Set Up Environment Variables

**Steps:**

1. **Open Cloud Shell:**
2. **Set AUTH_TOKEN:**
   - Replace `INSERT_SML_BEARER_TOKEN` with the token you copied earlier:
     ```bash
     export AUTH_TOKEN="INSERT_SML_BEARER_TOKEN"
     ```
3. **Download and Extract Lab Assets:**

4. **Set ENDPOINT Variable:**
   - Define the endpoint for predictions:
     ```bash
     export ENDPOINT="https://sml-api-vertex-kjyo252taq-uc.a.run.app/vertex/predict/tabular_classification"
     ```
5. **Set INPUT_DATA_FILE Variable:**
   - Define the input data file:
     ```bash
     export INPUT_DATA_FILE="INPUT-JSON"
     ```

6. **Review Lab Assets:**
   - **Files Overview:**
     - **INPUT-JSON:** Contains the data for making predictions.
     - **smlproxy:** Application used to communicate with the backend.

#### Step 6.2: Make a Prediction Request

**Steps:**

1. **Understand INPUT-JSON Structure:**
   - The `INPUT-JSON` file contains the following columns:
     - **age:** Age of the client.
     - **ClientID:** Unique identifier for the client.
     - **income:** Annual income of the client.
     - **loan:** Loan amount requested.

2. **Initial Prediction Request:**
   - Execute the following command to make a prediction:
     ```bash
     ./smlproxy tabular \
       -a $AUTH_TOKEN \
       -e $ENDPOINT \
       -d $INPUT_DATA_FILE
     ```
   - **Response:**
     ```bash
     SML Tabular HTTP Response:
     2022/01/10 15:04:45 {"model_class":"0","model_score":0.9999981}
     ```
   - **Interpretation:**
     - **model_class:** `0` indicates the prediction class (e.g., `0` for repay, `1` for default).
     - **model_score:** Confidence score of the prediction.

3. **Modify INPUT-JSON for a New Scenario:**
   - Edit the `INPUT-JSON` file to test a different loan scenario:
     ```bash
     nano INPUT-JSON
     ```
   - **Replace Content:**
     ```csv
     age,ClientID,income,loan
     30.00,998,50000.00,20000.00
     ```
   - **Save and Exit:**
 
4. **Make Another Prediction Request:**
   - Execute the prediction command again:
     ```bash
     ./smlproxy tabular \
       -a $AUTH_TOKEN \
       -e $ENDPOINT \
       -d $INPUT_DATA_FILE
     ```
   - **Response:**
     ```bash
     SML Tabular HTTP Response:
     2022/01/10 15:04:45 {"model_class":"0","model_score":1.0322887E-5}
     ```
   - **Interpretation:**
     - A low **model_score** indicates a high confidence in predicting that the person will **repay** the loan (`model_class: 0`).

#### Step 6.3: Customize Predictions

**Steps:**

1. **Create Custom Scenarios:**
   - Modify the `INPUT-JSON` file with different client profiles to see how the model responds.

2. **Automate Predictions:**
   - You can script multiple prediction requests by iterating over different input data files.

3. **Analyze Predictions:**
   - Use the prediction results to understand which client profiles are likely to repay loans and which are at risk of defaulting.

---

## Key Terminologies

- **Vertex AI:** Google's unified machine learning platform that enables building, deploying, and scaling ML models.
- **Classification:** A type of supervised learning where the model predicts categorical labels.
- **Regression:** A type of supervised learning where the model predicts continuous numerical values.
- **Precision:** The ratio of true positive predictions to the total predicted positives.
- **Recall:** The ratio of true positive predictions to the actual positives.
- **Confusion Matrix:** A table used to evaluate the performance of a classification model by comparing actual vs. predicted labels.
- **Feature Importance:** A metric that indicates how useful each feature was in the construction of the model.
- **Endpoint:** A deployed model's API interface that allows serving predictions.
- **Bearer Token:** An authentication token that grants access to protected resources.
- **cURL:** A command-line tool used to send HTTP requests, utilized here to interact with APIs.
- **Environment Variable:** Variables that are set in the operating system to pass configuration information to applications.
- **Model Score:** Confidence score indicating the probability associated with a prediction.
- **Salience:** A measure ranging from 0 to 1 indicating the importance of an entity within the context of the text.
- **Explainable AI:** A set of tools and frameworks to help understand and interpret predictions made by machine learning models.

---

## Conclusion

I successfully completed the **"Vertex AI: Predicting Loan Risk"** project. Here's a summary of what I accomplished:

- **Data Preparation:** Uploaded and prepared a tabular dataset for machine learning.
- **Model Training:** Utilized Vertex AI's AutoML to train a classification model predicting loan repayment risk.
- **Model Evaluation:** Understood how to evaluate model performance using metrics like precision, recall, and confusion matrix.
- **Model Deployment:** Learned the steps required to deploy a trained model to an endpoint for serving predictions.
- **Authentication:** Retrieved a Bearer Token to securely interact with the deployed model.
- **Predictions:** Made predictions using the deployed model via the SML service, interpreting the results to assess loan repayment risk.

---

## Future Work

To further enhance this project and my machine learning skills, I plan to:

1. **Enhance Model Complexity:**
   - Experiment with different budget allocations and training times to optimize model performance.
   - Explore feature engineering techniques to improve model accuracy.

2. **Real-time Predictions:**
   - Integrate the deployed model into a web application or service to provide real-time loan risk assessments.

3. **Model Monitoring:**
   - Implement monitoring to track model performance over time and detect any degradation or biases.

4. **Explore Other Vertex AI Features:**
   - Utilize **Custom Training** and **Hyperparameter Tuning** for more control over the model training process.
   - Explore **Explainable AI** to gain deeper insights into model decisions.

5. **Automate Workflows:**
   - Develop automated pipelines using **Vertex AI Pipelines** to streamline the model training and deployment process.

6. **Expand to Other ML Problems:**
   - Apply similar methodologies to different classification or regression problems, such as customer churn prediction or sales forecasting.

---

## Acknowledgements

- **Google Cloud Platform:** For providing robust and scalable machine learning tools.
- **Qwiklabs:** For offering hands-on labs that facilitate practical learning experiences.
- 
---

## License

This project is licensed under the [MIT License](LICENSE).

---

