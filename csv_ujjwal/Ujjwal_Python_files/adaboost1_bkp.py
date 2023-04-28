import numpy as np
import pandas as pd

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

train_values = pd.read_csv('train_values.csv',index_col='building_id')
train_labels = pd.read_csv('train_labels.csv',index_col='building_id')


print(train_values.dtypes)

print(train_values.apply(lambda col: col.unique()))


print(train_values.columns)


object_columns = train_values.select_dtypes(include=['object']).columns
print(object_columns)

for col in object_columns:
    train_values[col] = pd.factorize(train_values[col])[0]

print(train_values.apply(lambda col: col.unique()))
#train_values, train_labels = SMOTE().fit_resample(train_values, train_labels)


'''
selected_features = ['foundation_type',
                     'area_percentage',
                     'height_percentage',
                     'count_floors_pre_eq',
                     'land_surface_condition',
                     'has_superstructure_cement_mortar_stone']


train_values_subset = train_values[selected_features]

train_values_subset = pd.get_dummies(train_values_subset)
'''
base = DecisionTreeClassifier(max_depth=10)
pipe = make_pipeline(StandardScaler(),
                     AdaBoostClassifier(base_estimator=base,random_state=2018))
'''
param_grid = {'randomforestclassifier__n_estimators': [1, 20],
              'randomforestclassifier__min_samples_leaf': [1, 5]}
'''
param_grid = {'adaboostclassifier__n_estimators': [499, 500]}
gs = GridSearchCV(pipe, param_grid, cv=5,n_jobs=10)

gs.fit(train_values, train_labels.values.ravel())

print(gs.best_params_)
in_sample_preds = gs.predict(train_values)
print(f1_score(train_labels, in_sample_preds, average='micro'))

test_values = pd.read_csv('test_values.csv', index_col='building_id')
#test_values_subset = test_values[selected_features]
#test_values_subset = pd.get_dummies(test_values_subset)

print(test_values.apply(lambda col: col.unique()))


print(test_values.columns)


#object_columns = test_values.select_dtypes(include=['object']).columns
print(object_columns)

for col in object_columns:
    test_values[col] = pd.factorize(test_values[col])[0]

predictions = gs.predict(test_values)
submission_format = pd.read_csv('submission_format.csv', index_col='building_id')
my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)

my_submission.to_csv('submission.csv')

