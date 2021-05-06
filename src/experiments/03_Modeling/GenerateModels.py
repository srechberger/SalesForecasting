import os

# Build all Models
os.system('python 01_ETS/ETS_Store708.py')
os.system('python 01_ETS/ETS_Store198.py')
os.system('python 01_ETS/ETS_Store897.py')

os.system('python 02_SARIMA/SARIMA_Store708.py')
os.system('python 02_SARIMA/SARIMA_Store198.py')
os.system('python 02_SARIMA/SARIMA_Store897.py')

os.system('python 03_SVR/SVR.py')

os.system('python 04_K-NN/KNN.py')

os.system('python 05_FFNN/FFNN_Baseline.py')
os.system('python 05_FFNN/FFNN_GridSearch_708.py')
os.system('python 05_FFNN/FFNN_GridSearch_198.py')
os.system('python 05_FFNN/FFNN_GridSearch_897.py')

os.system('python 06_LSTM/LSTM.py')

os.system('python 07_XGBoost/XGBoost.py')

os.system('python 08_RandomForest/RandomForest.py')
