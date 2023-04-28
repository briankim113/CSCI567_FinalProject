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

import numpy
train_values = pd.read_csv('test_values.csv',index_col='building_id')
train_labels = pd.read_csv('train_labels.csv',index_col='building_id')


print(train_values.dtypes)

print(train_values.apply(lambda col: col.unique()))

train_values['area_pressure'] = (train_values['count_floors_pre_eq'] / train_values['area_percentage'])
train_values['old_height'] = (train_values['age'] * train_values['height_percentage'])
train_values['fam_weight'] = (train_values['count_families'] * train_values['area_pressure'])
train_values['fam_effect'] = (train_values['count_families'] * train_values['old_height'])
train_values['all_int_plus'] = (train_values['count_families'] + train_values['height_percentage']+ train_values['area_percentage']+train_values['count_floors_pre_eq']+train_values['age'])
train_values['all_int_mul'] = (train_values['count_families'] * train_values['height_percentage']* train_values['area_percentage']*train_values['count_floors_pre_eq']*train_values['age'])
#train_values['all_int_log'] = (np.log(train_values['count_families']) + np.log(train_values['height_percentage'])+ np.log(train_values['area_percentage'])+np.log(train_values['count_floors_pre_eq'])+np.log(train_values['age']))
train_values['all_int_sin'] = (np.sin(train_values['count_families']) + np.sin(train_values['height_percentage'])+ np.sin(train_values['area_percentage'])+np.sin(train_values['count_floors_pre_eq'])+np.sin(train_values['age']))
train_values['all_int_cos'] = (np.cos(train_values['count_families']) + np.cos(train_values['height_percentage'])+ np.cos(train_values['area_percentage'])+np.cos(train_values['count_floors_pre_eq'])+np.cos(train_values['age']))

print(train_values.head())
train_values.to_csv('test_feat.csv')
