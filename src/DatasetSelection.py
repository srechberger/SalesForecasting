# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 01 rossmann (get and prepare data for analysis)
rossmann_path = "../../../data/rossmann/input/train.csv"
rossmann_data = pd.read_csv(rossmann_path)
# Parse date column from object to datetime
rossmann_data['Date'] = pd.to_datetime(rossmann_data['Date'], format="%Y-%m-%d")

# 02 xxx (get and prepare data for analysis)

# 03 xxx (get and prepare data for analysis)



# Funktion für Variationskoeffizient (cv = koe)
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

# Berechnung Variationskoeffizient pro Store
rows = []
for i in range(1, 1116):
    store_data = sales.loc[sales.Store == i]['Sales']
    koe = cv(store_data)
    rows.append([i, koe])

koe_store = pd.DataFrame(rows, columns=["StoreId", "CV"])
koe_store = koe_store.sort_values(by='CV', ascending=False)