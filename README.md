🏡 Home Loan Prediction

This project aims to predict the approval status of home loan applications using machine learning techniques. 
By analyzing applicant data, the model determines the likelihood of loan approval, assisting financial institutions in making informed decisions.

📂 Project Structure

- data/: Contains the dataset used for training and evaluation.
- notebooks/: Jupyter notebooks detailing the exploratory data analysis (EDA) and model development processes.
- app.py: Flask web application for user interaction with the prediction model.
- best_model.pkl: Serialized version of the trained model for deployment.
- requirements.txt: Lists all Python dependencies required to run the project.
- run_training.py: Script to initiate model training.
- train_model.py: Contains functions and classes related to model training and evaluation.

🛠️ Installation

1. Clone the Repository:

   ```bash
   git clone https://github.com/Shivam-809/home_loan_prediction.git

2. Create a Virtual Environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

3. Install Dependencies:
pip install -r requirements.txt

4. Train the Model:
python run_training.py

5. Run the Flask Application:
streamlit run app.py
Ensure that streamlit is installed in your system 

📝 Data Description:

The dataset includes the following features:

ApplicantIncome: The income of the applicant.

CoapplicantIncome: The income of the co-applicant.

LoanAmount: The loan amount requested.

Loan_Amount_Term: Term of the loan in months.

Credit_History: Credit history of the applicant (1: good, 0: bad).

Gender: Gender of the applicant.

Married: Marital status.

Dependents: Number of dependents.

Education: Education level of the applicant.

Self_Employed: Employment status.

Property_Area: The area type where the property is located (Urban/Semiurban/Rural).
