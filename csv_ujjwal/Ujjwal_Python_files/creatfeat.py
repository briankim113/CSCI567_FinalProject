import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

# the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# for combining the preprocess with model training
from sklearn.pipeline import make_pipeline

# for optimizing the hyperparameters of the pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from pycaret.datasets import get_data

from pycaret.classification import *

import numpy
train_values = pd.read_csv('train_values.csv',index_col='building_id')
train_labels = pd.read_csv('train_labels.csv',index_col='building_id')
print(train_labels['damage_grade'])

train_values['damage_grade']=train_labels['damage_grade']

print(train_values.head())
#train_values = train_values[['has_superstructure_mud_mortar_stone', 'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_brick', 'has_superstructure_stone_flag', 'geo_level_1_id', 'age', 'foundation_type', 'roof_type', 'count_floors_pre_eq', 'has_superstructure_cement_mortar_brick', 'other_floor_type', 'has_superstructure_timber', 'count_families', 'geo_level_2_id', 'has_superstructure_rc_engineered', 'has_superstructure_bamboo', 'position', 'has_secondary_use_other', 'ground_floor_type','damage_grade']]
#train_values = train_values[['has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'geo_level_1_id', 'foundation_type', 'roof_type', 'has_superstructure_cement_mortar_brick', 'other_floor_type', 'ground_floor_type','damage_grade']]
clf1 = setup(data = train_values, target = 'damage_grade')

#best_model = compare_models(exclude=['xgboost'], fold=5)

xgboost = create_model('lightgbm')
#tuned_xgboost = tune_model(xgboost, n_iter=5)
#plot_model(xgboost, plot = 'feature')
test_values = pd.read_csv('test_values.csv',index_col='building_id')
#test_values = test_values[['has_superstructure_mud_mortar_stone', 'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_brick', 'has_superstructure_stone_flag', 'geo_level_1_id', 'age', 'foundation_type', 'roof_type', 'count_floors_pre_eq', 'has_superstructure_cement_mortar_brick', 'other_floor_type', 'has_superstructure_timber', 'count_families', 'geo_level_2_id', 'has_superstructure_rc_engineered', 'has_superstructure_bamboo', 'position', 'has_secondary_use_other', 'ground_floor_type']]
#test_values=test_values[['has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'geo_level_1_id', 'foundation_type', 'roof_type', 'has_superstructure_cement_mortar_brick', 'other_floor_type', 'ground_floor_type']]
print(test_values.head())
xgfinal = finalize_model(xgboost)

predictions = predict_model(xgfinal)

pred_test = predict_model(xgfinal,test_values)
print(pred_test.head())
submission_format = pd.read_csv('submission_format.csv', index_col='building_id')
my_submission = pd.DataFrame(data=pred_test['prediction_label'].values.tolist(),
                             columns=submission_format.columns,
                             index=submission_format.index)

my_submission.to_csv('submission_pycaret.csv')
plot_model(xgboost, plot = 'feature')
