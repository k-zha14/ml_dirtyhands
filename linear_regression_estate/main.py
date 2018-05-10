import module_kit
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

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
    sset.drop(["income_cat"], axis=1, inplace=True)
# backup
housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population',
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True
             )  # alpha control the color density of scatter dot
# principle component analysis, calculate the correlation coefficient
corr_matrix = housing.corr()  # return Pearson's r matrix
print(corr_matrix['median_house_value'].sort_values(ascending=False))  # descending order

# plt.show()



