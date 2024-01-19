# Bayesian Network for Heart Disease Prediction
# Probabilistic Inference using pgmpy

#Install pgmpy if not already installed
!pip install pgmpy

import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load the Heart Disease dataset
heartDisease = pd.read_csv('HeartDisease.csv')
heartDisease = heartDisease.replace('?', np.nan)

# Display sample instances from the dataset
print('Sample instances from the dataset are given below:')
print(heartDisease.head())

# Display attributes and datatypes
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Define the Bayesian Network structure
model = BayesianNetwork([('age', 'heartdisease'), ('gender', 'heartdisease'), ('exang', 'heartdisease'),
                         ('cp', 'heartdisease'), ('restecg', 'heartdisease'), ('chol', 'heartdisease')])

# Learn CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
HeartDiseaseTestInfer = VariableElimination(model)

# Probability of Heart Disease given evidence=restecg
q1 = HeartDiseaseTestInfer.query(variables=['heartdisease'], evidence={'restecg': 1})
print('\n1. Probability of HeartDisease given evidence=restecg:')
print(q1)

# Probability of Heart Disease given evidence=cp
q2 = HeartDiseaseTestInfer.query(variables=['heartdisease'], evidence={'cp': 2})
print('\n2. Probability of HeartDisease given evidence=cp:')
print(q2)