from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Get Training Data
# Store 198
X_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198_y.pkl')

# Get Test Data
# Store 198
X_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_3M.pkl')
y_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_3M.pkl')

# Drop not important features
X_train_198 = X_train_198.drop([
    'Store',
    'StoreType',
    'Assortment',
    'CompetitionDistance',
    'Promo2',
    'DayOfMonth',
    'IsPromoMonth'],
    axis=1)

X_test_3M_198 = X_test_3M_198.drop([
    'Store',
    'StoreType',
    'Assortment',
    'CompetitionDistance',
    'Promo2',
    'DayOfMonth',
    'IsPromoMonth'],
    axis=1)

# Scale values (performance relevant)
scaler = MinMaxScaler()
X_train_198[X_train_198.columns] = scaler.fit_transform(X_train_198[X_train_198.columns])
X_test_3M_198[X_test_3M_198.columns] = scaler.transform(X_test_3M_198[X_test_3M_198.columns])


# Function to build the FFNN model with a variable number of nodes
def build_model(neurons):
    model = Sequential()
    model.add(Dense(neurons, input_shape=[10], activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Construct the Regressor model
regressor = KerasRegressor(build_fn=build_model, verbose=0)
parameters = {'batch_size': [5, 10, 20],
              'epochs': [300],
              'neurons': [20, 25, 30, 40]}

grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=10)

# Fit the various models using the object defined above
grid_search = grid_search.fit(X_train_198, y_train_198)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print('Best Parameters:', best_parameters)
# Best Parameters: {'batch_size': 5, 'epochs': 300, 'neurons': 40}

# -----------------------------------------------------------------------------------------------
# Build model with best params of GridSearchCV

# Now we build and train the optimal model
ffnn_model = build_model(neurons=best_parameters['neurons'])

# Store the history of loss of the optimal function
history = ffnn_model.fit(X_train_198, y_train_198, epochs=best_parameters['epochs'],
                         batch_size=best_parameters['batch_size'], verbose=0)

# Save the model
ffnn_model.save('../../04_Evaluation/00_Models/ffnn_model_198_gs')

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.plot()
path = "../../../../data/rossmann/output/ffnn_learning_curve_198"
plt.savefig(path)
# plt.show()
