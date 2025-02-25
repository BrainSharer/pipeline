import os
import sys
import xgboost as xgb
import pandas as pd
from compress_pickle import dump
import pickle

model_file = 'models_example.pkl'
if os.path.exists(model_file):
    print(f'FOUND MODEL FILE @ {model_file}')
else:
    print(f'MODEL FILE NOT FOUND @ {model_file}')
    sys.exit()

# Load the model from the pickle file
with open(model_file, 'rb') as f:
    models = pickle.load(f)

print(f'LOADED MODEL FROM {model_file} type: {type(models)}')

for i, model in enumerate(models):
    if isinstance(model, xgb.Booster):
        new_model = f'model_{i}.json'
        model.save_model(new_model)
        print(f'MODEL {i} SAVED TO {new_model}')
        print(f'MODEL {i} IS AN XGBOOST BOOSTER')
    else:
        print(f'MODEL {i} IS NOT AN XGBOOST BOOSTER type: {type(model)}')



new_model_array = []
for i in range(len(models)):
    bst=model[i]
    #updated_model = xgb.Booster(model_file=bst)
    model_file = os.path.join(os.getcwd(), f'model_{i}.json')
    with open(model_file, 'rb') as f:
        #loaded_model = xgb.XGBClassifier()
        loaded_model = xgb.Booster()
        loaded_model.load_model(model_file)

        new_model_array.append(loaded_model)

#Save the model with the new version (2.0.2)
new_model_file = os.path.join(os.getcwd(), 'models_example_json.pkl')
dump(new_model_array, new_model_file)
print(f"Model successfully saved as JSON to {new_model_file}")

with open(new_model_file, 'rb') as f:
    models = pickle.load(f)

for i, model in enumerate(models):
    # Check if the model is an XGBoost Booster
    if isinstance(model, xgb.Booster):
        print(f"The loaded {i} model is an XGBoost Booster.")
    else:
        print(f"The loaded {i} model is NOT an XGBoost Booster. type: {type(model)}")

jsons = os.listdir(os.getcwd())
jsons = [j for j in jsons if j.endswith('.json')]
for j in jsons:
    os.remove(j)
    print(f"Removed {j}")