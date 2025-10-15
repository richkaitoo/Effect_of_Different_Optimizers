import os
import pandas as pd
import kagglehub
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    data_path = kagglehub.dataset_download('dhirajnirne/california-housing-data', path=data_dir)
    housing = pd.read_csv(os.path.join(data_path, 'housing.csv'))
    return housing

def preprocess_housing_data(housing):
    # Separating numerical and categorical columns
    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_cat = housing[["ocean_proximity"]]
    
    # Imputing missing values in numerical columns
    imputer = SimpleImputer(strategy="median")
    housing_num_tr = pd.DataFrame(imputer.fit_transform(housing_num), columns=housing_num.columns, index=housing_num.index)
    
    # One-hot encode categorical columns
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    # Scaling numerical columns
    scaler = StandardScaler()
    housing_num_scaled = scaler.fit_transform(housing_num_tr)
    housing_num_scaled = pd.DataFrame(housing_num_scaled, columns=housing_num_tr.columns.astype(str), index=housing_num_tr.index)
    
    # Combining the preprocessed data
    housing_preprocessed = pd.concat([housing_num_scaled, pd.DataFrame(housing_cat_1hot.toarray(), index=housing_cat.index)], axis=1)
    housing_preprocessed.columns = housing_preprocessed.columns.astype(str)
    return housing_preprocessed

def stratified_split(housing):
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    strat_splits = []
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])
    return strat_splits