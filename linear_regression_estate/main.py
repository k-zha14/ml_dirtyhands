import module_kit
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# fetch the estate data from github
# module_kit.fetch_housing_data()
# read csv
housing = module_kit.load_housing_data()
# plot to see the big picture
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()
# stratify the sample
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)  # ceil the max value at 5
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# stratify sampling the train and test set
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# delete column income_car
for sset in (strat_train_set, strat_test_set):
    sset.drop(["income_cat"], axis=1, inplace=True)  # focus on the inplace, no copy
# backup
housing = strat_train_set.copy()
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
#              s=housing['population']/100, label='population',
#              c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True
#              )  # alpha control the color density of scatter dot
# produce some combination features
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
# principle component analysis, calculate the correlation coefficient
# corr_matrix = housing.corr()  # return Pearson's r
# print(corr_matrix["median_house_value"].sort_values(ascending=False))   # descending order
# separate the labels from training data_set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()  # training labels
housing_num = housing.drop("ocean_proximity", axis=1)  # return all numerical column
housing_cat = housing["ocean_proximity"]


# data cleaning
# transfer pandas.dataframe to np.array
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# fix NA
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
# numerical features: fix NA, feature scaling
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
# category feature: one-hot encoder
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    # ('label_binarizer', LabelBinarizer()),
])
# emerge together
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
housing_prepared = full_pipeline.fit_transform(housing)
# one-hot encoder
housing_prepared = np.delete(housing_prepared, -1, 1)
housing_cat_1hot = LabelBinarizer().fit_transform(housing_cat)
housing_prepared = np.c_[housing_prepared, housing_cat_1hot]
# train model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#test
little_data = housing_prepared[:5]
little_labels = housing_labels.iloc[:5]
print("Prediction:\t", lin_reg.predict(little_data))
print("labels:\t", list(little_labels))
housing_prediction = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_prediction)
print(lin_mse)
# plt.show()



