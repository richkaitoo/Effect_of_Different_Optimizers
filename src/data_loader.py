import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cities = {
            'los_angeles': (-118.2437, 34.0522),
            'san_francisco': (-122.4194, 37.7749)
        }
        self.earth_radius = 6371   

    def fit(self,  X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Calculating new features
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        X['population_per_household'] = X['population'] / X['households']
        
        # Calculating distance to each city
        for city, (lon, lat) in self.cities.items():
            X[f'distance_to_{city}'] = self.haversine_distance(X, lon, lat)
        return X

    def haversine_distance(self, X, lon2, lat2):
        lon1, lat1 = X['longitude'], X['latitude']
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return self.earth_radius * c

def create_new_features(data):
    data['rooms_per_household'] = data['total_rooms'] / data['households']
    data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
    data['population_per_household'] = data['population'] / data['households']
    return data