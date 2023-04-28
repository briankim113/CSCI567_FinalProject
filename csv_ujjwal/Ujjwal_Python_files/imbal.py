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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import f1_score

train_values = pd.read_csv('train_values.csv',index_col='building_id')
train_labels = pd.read_csv('train_labels.csv',index_col='building_id')


print(train_values.dtypes)

print(train_values.apply(lambda col: col.unique()))


print(train_values.shape)
print(train_labels.shape)
train_values['damage_grade'] = train_labels['damage_grade']
print(train_values.shape)
print(train_values)
print(len(train_values[train_values.damage_grade==1]))
print(len(train_values[train_values.damage_grade==2]))
print(len(train_values[train_values.damage_grade==3]))

train_values=train_values.drop(columns=['damage_grade'])
object_columns = train_values.select_dtypes(include=['object']).columns
print(object_columns)

for col in object_columns:
    train_values[col] = pd.factorize(train_values[col])[0]

print(train_values.apply(lambda col: col.unique()))
oversample = RandomOverSampler(sampling_strategy='not majority')
train_values, train_labels = oversample.fit_resample(train_values, train_labels)
print(train_labels.value_counts())


pipe = make_pipeline(StandardScaler(),
                     RandomForestClassifier(random_state=2018))

'''
param_grid = {'randomforestclassifier__n_estimators': [200],
              'randomforestclassifier__min_samples_leaf': [1, 5]}
'''
param_grid = {'randomforestclassifier__bootstrap': [True, False],
 'randomforestclassifier__max_depth': [10, 20, 30, None],
 'randomforestclassifier__max_features': ['sqrt'],
 'randomforestclassifier__min_samples_leaf': [1, 2],
 'randomforestclassifier__min_samples_split': [5, 10],
 'randomforestclassifier__n_estimators': [100]}
gs = GridSearchCV(pipe, param_grid,cv=5,n_jobs=10,verbose=2)

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
