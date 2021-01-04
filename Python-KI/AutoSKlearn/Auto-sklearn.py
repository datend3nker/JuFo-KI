from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import yaml
import autosklearn.classification

def saveDump(modelName, dump, file):
    __model_log ={}
    with open(str(file), 'a') as f:
        __model_log[modelName] = dump
        yaml.dump(__model_log, f, sort_keys=False)
    return __model_log

data = pd.read_csv('./questions.csv')
data_target = data['is_duplicate'].values.astype(bool)
data_value = data['question1'].astype(str) + " " + data['question2'].astype(str)

Transform = Pipeline(verbose = 50, steps=[
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])

transdata = Transform.fit_transform(data_value)
X_train, X_test, y_train, y_test = train_test_split(transdata, data_target, test_size=0.25, random_state=42)


automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=34200,
                per_run_time_limit=1000,
                ml_memory_limit=30000,
                n_jobs=-1,
                #initial_configurations_via_metalearning= False,
                ensemble_memory_limit=2000,
                #resampling_strategy='cv',
                #resampling_strategy_arguments={'folds': 5}
            )

automl.fit(X_train, y_train, dataset_name='QuestionPairs')
#automl.refit(X_train, y_train)
predictions = automl.predict(X_test)
print("=================================================================================================================================================================================")
#print((model_process['Auto-sklearn']).show_models())
print(automl.show_models())
print("=================================================================================================================================================================================")
#print(model_process['Auto-sklearn'].sprint_statistics())
print(automl.sprint_statistics())
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
print("AUC:", sklearn.metrics.auc(y_test, predictions))
print("=================================================================================================================================================================================")
#dump = (model_process['Auto-sklearn']).show_models()
dump = (automl.show_models()), "\n\n", automl.sprint_statistics()
with open("/mnt/c/Users/ludwi/source/repos/JugendForscht/Results/AutoSklearn.yaml", 'w') as f:
        yaml.dump(dump, f, sort_keys=False)
