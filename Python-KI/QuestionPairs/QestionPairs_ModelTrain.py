from sklearn.pipeline import Pipeline
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import datetime
from joblib import dump, load
from sklearn.preprocessing import MaxAbsScaler
"""
    Text-klassifikations-Algorithmen
    Random Forest
    Support Vector Machine
    K Nearest Neighbors
    Multinomial Naïve Bayes
    Multinomial Logistic Regression
    Gradient Boosting
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier

model_list = {
    "RandomForestClassifier": RandomForestClassifier(n_jobs = -1, verbose = 50), 
    "svm.SVC": svm.SVC(cache_size = 1000, verbose = 50, max_iter = 1000), 
    "svm.LinearSVC": svm.LinearSVC(verbose = 50),
#    "svm.NuSVC": svm.NuSVC(cache_size = 1000, verbose = 50, max_iter = 100, nu = 0.5),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs = -1), 
    "MultinomialNB": MultinomialNB(), 
    "LogisticRegression": LogisticRegression(n_jobs = -1, verbose = 50, solver = 'saga'), 
    "GradientBoostingRegressor": GradientBoostingRegressor(verbose = 50),
    "SGDClassifier": SGDClassifier(n_jobs = -1, verbose = 50),}

def Filename():
    __name = "ModelResult_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + ".yaml"
    return __name

def dumpResult(score, time, modelparm,):
    __pipestep = ""
    __akt_model = {
            "Zeit": None, "Leistung": None, "Parameter": None,}
    __akt_model["Zeit"] = str(time)
    __akt_model["Leistung"] = str(score)
    for i in range(len(modelparm['steps'])):
        __pipestep = __pipestep + str(modelparm['steps'][i])+ " "
    modelparm['steps'] = __pipestep
    for x in modelparm:
        if modelparm[x] is not (str or int or float or bool or None or complex):
            modelparm[x] = str(modelparm[x])
    __akt_model['Parameter'] = modelparm
    return(__akt_model)
    
def saveDump(modelName, dump, file):
    __model_log ={}
    with open(str(file), 'a') as f:
        __model_log[modelName] = dump
        yaml.dump(__model_log, f, sort_keys=False)
    return __model_log

def saveModel(modelName, estmodel):
    __filename = str(modelName) + ".joblib"
    dump(estmodel, __filename)

#laed den Datensatz un konvertiert die "Labels" zu looleans. Zum Schluss werden die 2 Strings zusammen 
#gefügtm um in Features transformiert zu werden 
data = pd.read_csv('./questions.csv')
data_target = data['is_duplicate'].values.astype(bool)
data_value = data['question1'].astype(str) + " " + data['question2'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(data_value, data_target, test_size=0.25, random_state=42)

acFile = Filename()

for ModelName, model in model_list.items():
        start = datetime.datetime.now()
        model_process = Pipeline(verbose = 50, steps=[
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('model', model),
            ])

        print("beginne Training mit " + ModelName)
        model_process.fit(X_train, y_train)
        end = datetime.datetime.now()
        print("Das Model ist ", model_process.score(X_test, y_test),"% genau")
        x = dumpResult(model_process.score(X_test, y_test), (end - start), model_process.get_params())
        saveDump(ModelName, x, acFile)
        saveModel(ModelName, model)