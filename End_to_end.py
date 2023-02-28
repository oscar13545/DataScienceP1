from sklearn.metrics import mean_squared_error
import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import joblib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path=HOUSING_PATH):
  os.makedirs(housing_path, exist_ok=True)
  tgz_path = os.path.join(housing_path, "housing.tgz")
  urllib.request.urlretrieve(housing_url, tgz_path)
  housing_tgz = tarfile.open(tgz_path)
  housing_tgz.extractall(path=housing_path)
  housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
  csv_path = os.path.join(housing_path, "housing.csv")
  return pd.read_csv(csv_path)

fetch_housing_data(HOUSING_URL, HOUSING_PATH)
housing = load_housing_data()
housing.head()

#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print(len(train_set))
#print(len(test_set))

housing["income_cat"] = pd.cut(housing["median_income"],
                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                        labels=[1, 2, 3, 4, 5])

#housing["income_cat"].hist() #Muestra la tablas

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

#housing.plot(kind="scatter", x="longitude", y= "latitude", alpha=0.1)
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#s=housing["population"]/100, label="population", figsize= (10,7),
# c="median house value", cmap=plt.get_cmap("jet"), colorbar=True,)

#plt.legend()


#attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
#scatter_matrix(housing[attributes], figsize=(12, 8))

housing ["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing ["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing[ "households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#print(len(housing_labels))
#print(len(housing))

imputer = SimpleImputer(strategy="median" )
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
x = imputer.transform(housing_num)
housing_tr = pd.DataFrame(x, columns=housing_num.columns, index=housing_num. index)
#print(housing_tr)

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

#ordinal_encoder.categories_


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
#housing_cat_1hot.toarray()
#print(housing_cat_1hot)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, add_bedrooms_per_room = True): 
    self.add_bedrooms_per_room = add_bedrooms_per_room
  def fit(self, x, y=None):
    return self 
  def transform(self, x):
    rooms_per_household = x[:, rooms_ix] / x[:, households_ix]
    population_per_household = x[:, population_ix] / x[:, households_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
      return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
      return np.c_[x, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
#print(housing_extra_attribs)



num_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy="median")),
  ('attribs_adder', CombinedAttributesAdder()),
  ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), 
("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape

joblib.dump(full_pipeline, "full_pipeline.pkl")

#-------------------------------------------------
#lin_reg = LinearRegression()
#lin_reg.fit(housing_prepared, housing_labels)
#-------------------------------------------------
#-------------------------------------------------
#tree_reg = DecisionTreeRegressor ()
#tree_reg.fit (housing_prepared, housing_labels)
#-------------------------------------------------


#scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean ())
  print("Standard deviation:", scores.std())
  print("---------------------------------------------")

#display_scores(tree_rmse_scores)

#lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#lin_rmse_scores = np.sqrt(-lin_scores)
#display_scores(lin_rmse_scores)


#forest_reg = RandomForestRegressor ()
#forest_reg.fit(housing_prepared, housing_labels)

#forest_rmse = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#forest_rmse_scores = np.sqrt(-forest_rmse)
#display_scores(forest_rmse_scores)

#-------------------------------------------------------
# RandomForestRegressor

""""
param_grid = [
  {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
  {'bootstrap': [False], 'n_estimators': [3, 10],'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(np.sqrt(-mean_score), params)


final_model = grid_search.best_estimator_
joblib.dump(final_model, "final_model.pkl")

final_model = joblib.load("final_model.pkl")

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("------------------------------------------------------")
print("Evaluación del Random Forest Regressor: ",final_rmse)


confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
resultado=np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
loc=squared_errors.mean (), scale=stats.sem(squared_errors)))
print("Resultado del Status del Random Forest Regressor: ",resultado)
"""
#---------------------------------------------------------
#       -LinearRegression-
"""
param_gridLR = [{'fit_intercept': [True, False], 'positive' : [True, False]}]
lin_reg = LinearRegression()

grid_searchLR = GridSearchCV(lin_reg, param_gridLR, cv=5, 
                           scoring='neg_mean_squared_error',
                          return_train_score=True)

grid_searchLR.fit(housing_prepared, housing_labels)

cvres = grid_searchLR.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(np.sqrt(-mean_score), params)

final_modelLR = grid_searchLR.best_estimator_


joblib.dump(final_modelLR, "final_modelLR.pkl")
"""
final_modelLR = joblib.load("final_modelLR.pkl")

X_testLR = strat_test_set.drop("median_house_value", axis=1)
y_testLR = strat_test_set["median_house_value"].copy()

X_test_preparedLR = full_pipeline.transform(X_testLR)
final_predictionsLR = final_modelLR.predict(X_test_preparedLR)

final_mseLR = mean_squared_error(y_testLR, final_predictionsLR)
final_rmseLR = np.sqrt(final_mseLR)
print("------------------------------------------------------")
print("Evaluación del Linear Regression: ",final_rmseLR)

confidence = 0.95
squared_errorsLR = (final_predictionsLR - y_testLR) ** 2
resultadoLR=np.sqrt(stats.t.interval(confidence, len(squared_errorsLR) - 1,
loc=squared_errorsLR.mean (), scale=stats.sem(squared_errorsLR)))
print("Resultado del Status del Linear Regression: ",resultadoLR)


#------------------------------------------------------------------------
# DecisionTreeRegressor
"""
param_gridDTR = [{'criterion': ["friedman_mse", "squared_error", "absolute_error", "poisson"], 'splitter': ["best","random"], 'max_depth': [1,2,4]}]
tree_reg = DecisionTreeRegressor ()

grid_searchDTR = GridSearchCV(tree_reg, param_gridDTR, cv=5, 
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_searchDTR.fit(housing_prepared, housing_labels)

cvres = grid_searchDTR.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
 # print(np.sqrt(-mean_score), params)

final_modelDTR = grid_searchDTR.best_estimator_
joblib.dump(final_modelDTR, "final_modelDTR.pkl")
"""
final_modelDTR = joblib.load("final_modelDTR.pkl")

X_testDTR = strat_test_set.drop("median_house_value", axis=1)
y_testDTR = strat_test_set["median_house_value"].copy()

X_test_preparedDTR = full_pipeline.transform(X_testDTR)
final_predictionsDTR = final_modelDTR.predict(X_test_preparedDTR)

final_mseDTR = mean_squared_error(y_testDTR, final_predictionsDTR)
final_rmseDTR = np.sqrt(final_mseDTR)
print("------------------------------------------------------")
print("Evaluación del Decision Tree Regressor: ",final_rmseDTR)

confidence = 0.95
squared_errorsDTR = (final_predictionsDTR - y_testDTR) ** 2
resultadoDTR=np.sqrt(stats.t.interval(confidence, len(squared_errorsDTR) - 1,
loc=squared_errorsDTR.mean (), scale=stats.sem(squared_errorsDTR)))
print("Resultado del Status del Decision Tree Regressor: ",resultadoDTR)
#-----------------------------------------------------------------------
# Support Vector Machine
from sklearn.svm import SVR
"""
#param_gridDSVM = [{'kernel': ["linear", "poly"], 'gamma': ["scale","auto"], 'shrinking': [True, False]}]
#suppor_vector = SVR()

#grid_searchSVM = GridSearchCV(suppor_vector, param_gridDSVM, cv=5, 
 #                          scoring='neg_mean_squared_error',
  #                         return_train_score=True)

#grid_searchSVM.fit(housing_prepared, housing_labels)

#cvres = grid_searchSVM.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  #print(np.sqrt(-mean_score), params)
  
#final_modelSVM = grid_searchSVM.best_estimator_
"""
#joblib.dump(final_modelSVM, "final_modelSVM.pkl")
final_modelSVM = joblib.load("final_modelSVM.pkl")

X_testSVM = strat_test_set.drop("median_house_value", axis=1)
y_testSVM = strat_test_set["median_house_value"].copy()

X_test_preparedSVM = full_pipeline.transform(X_testSVM)
final_predictionsSVM = final_modelSVM.predict(X_test_preparedSVM)

final_mseSVM = mean_squared_error(y_testSVM, final_predictionsSVM)
final_rmseSVM = np.sqrt(final_mseSVM)
print("------------------------------------------------------")
print("Evaluación del Support Vector Machine: ",final_rmseSVM)

confidence = 0.95
squared_errorsSVM = (final_predictionsSVM - y_testSVM) ** 2
resultadoSVM = np.sqrt(stats.t.interval(confidence, len(squared_errorsSVM) - 1,
loc=squared_errorsSVM.mean (), scale=stats.sem(squared_errorsSVM)))
print("Resultado del Status del Support Vector Machine: ",resultadoSVM)
print("------------------------------------------------------")






