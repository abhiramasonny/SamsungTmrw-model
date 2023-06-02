
# Load the model from the saved .pt file
import torch
import pandas as pd
loaded_model = torch.load('heart_model.pt')
print("Heart model loaded from heart_model.pt")
"""
About this dataset

Age : Age of the patient

Sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

ca: number of major vessels (0-3)

cp : Chest Pain type chest pain type

Value 1: typical angina Value 2: atypical angina Value 3: non-anginal pain Value 4: asymptomatic trtbps : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg : resting electrocardiographic results

Value 0: normal Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria thalach : maximum heart rate achieved

target : 0= less chance of heart attack 1= more chance of heart attack



"""
data_test = {
    'age': 13,
    'sex': 0,
    'cp': 0,
    'trtbps': 130,
    'chol': 250,
    'restecg': 0,
    'thalachh': 150,
    'exng': 0,
    'oldpeak': 2.5,
    'slp': 1,
    'caa': 0,
    'thall': 2
}
data_test2 = {
    'age': 13,
    'sex': 0,
    'cp': 0,
    'trtbps': 130,
    'chol': 250,
    'restecg': 0,
    'thalachh': 150,
    'exng': 0,
    'oldpeak': 2.5,
}
# Convert the dictionary to a DataFrame
df_test = pd.DataFrame(data_test, index=[0])

# Make predictions using the loaded model
predictions = loaded_model.predict(df_test)

# Print the predicted result
if predictions[0] == 0:
    print("The model predicts that the person does not have a heart condition.")
else:
    print("The model predicts that the person has a heart condition.")
