import pandas as pd

# set display options
pd.set_option('display.max_columns', 15)

sales = pd.read_pickle('../../../data/rossmann/intermediate/sales.pkl')

print(sales.dtypes)

### Step 3: Select a model

### Step 4: Train the model

### Step 5: Evaluate the model

### Step 6: Tune parameters

### Step 7: Get predictions