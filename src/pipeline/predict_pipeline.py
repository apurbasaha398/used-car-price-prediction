import sys
import pandas as pd
from src.exception import CustomException
import dill as pickle
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifact", "model.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model.predict(features)
        except Exception as e:
            raise CustomException(e, sys)
    
class CustomData:
    def __init__(self,
        back_legroom: float,
        body_type: str,
        num_of_cylinders: int,
        exterior_color_grouped: str,
        front_legroom: float,
        fuel_tank_volume: float,
        fuel_type: str,
        horsepower: float,
        interior_color_grouped: str,
        make_name: str,
        maximum_seating: int,
        mileage: str,
        model_name: str,
        seller_rating: str,
        torque: float,
        transmission: str,
        wheel_system: str,
        wheelbase: float,
        volume: float,
        year: int,
        combined_fuel_economy: float,
        dealer_region: str):
        
        self.back_legroom = back_legroom
        self.body_type = body_type
        self.num_of_cylinders = num_of_cylinders
        self.exterior_color_grouped = exterior_color_grouped
        self.front_legroom = front_legroom
        self.fuel_tank_volume = fuel_tank_volume
        self.fuel_type = fuel_type
        self.horsepower = horsepower
        self.interior_color_grouped = interior_color_grouped
        self.make_name = make_name
        self.maximum_seating = maximum_seating
        self.mileage = mileage
        self.model_name = model_name
        self.seller_rating = seller_rating
        self.torque = torque
        self.transmission = transmission
        self.wheel_system = wheel_system
        self.wheelbase = wheelbase
        self.volume = volume
        self.year = year
        self.combined_fuel_economy = combined_fuel_economy
        self.dealer_region = dealer_region
        
    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                'back_legroom (in)': [self.back_legroom],
                'body_type': [self.body_type],
                'num_of_cylinders': [self.num_of_cylinders],
                'exterior_color_grouped': [self.exterior_color_grouped],
                'front_legroom (in)': [self.front_legroom],
                'fuel_tank_volume (gallon)': [self.fuel_tank_volume],
                'fuel_type': [self.fuel_type],
                'horsepower': [self.horsepower],
                'interior_color_grouped': [self.interior_color_grouped],
                'make_name': [self.make_name],
                'maximum_seating': [self.maximum_seating],
                'mileage': [self.mileage],
                'model_name': [self.model_name],
                'seller_rating': [self.seller_rating],
                'torque (rpm)': [self.torque],
                'transmission': [self.transmission],
                'wheel_system': [self.wheel_system],
                'wheelbase (in)': [self.wheelbase],
                'volume': [self.volume],
                'year': [self.year],
                'combined_fuel_economy': [self.combined_fuel_economy],
                'dealer_region': [self.dealer_region]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        