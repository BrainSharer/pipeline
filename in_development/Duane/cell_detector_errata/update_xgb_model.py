import os
from pathlib import Path
import xgboost as xgb
import joblib
import pandas as pd
from compress_pickle import dump, load

model_file = Path(os.getcwd(), 'models_example.pkl')
print(f'FOUND MODEL FILE @ {model_file}')

# Load the old model (1.6) - contains 30 models
model = pd.read_pickle(model_file)

new_model_array = []
for i in range(len(model)):
    bst=model[i]
    updated_model = xgb.Booster(model_file=bst)
    new_model_array.append(updated_model)

#Save the model with the new version (2.0.2)
new_model_file = Path(os.getcwd(), f'models_example_new.pkl')
dump(new_model_array, new_model_file.as_posix())