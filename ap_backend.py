## TODO optimize predict method via numpy calculation
## TODO add pruning for accuracy enhancement
# Disable Tensorflow Warnings/Errors
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class AnxietyPredictor:
    """An AI that can predict anxiety levels, according to the GAD-7 scale, based
    on leading factors of anxiety."""
    
    model_name = "anxiety_predictor"
    model_suffix = ".keras"
    model_directory = os.path.join(os.getcwd(), "static/model/") # model cant load without trailing slash
    
    def __init__(self, csv_filename="stress_level_dataset.csv"):
        # preparation stage
        self.raw_ai_dataset = self._load_dataset(csv_filename)
        self.features, self.label = self._get_data_points(self.raw_ai_dataset)
        
        # preprocessing stage
        self.normalized_ai_dataset, self.normalizer = self._normalize_dataset(self.raw_ai_dataset)
        self.nn_dataset = self._prepare_dataset(self.normalized_ai_dataset)
    
    
    def _load_dataset(self, csvfilename) -> Optional[pd.DataFrame]:
        """Loads a (CSV) dataset used to train the AI model by its filename
        Returns a Pandas DataFrame of the dataset when successfully loaded
        Returns None when the path to the dataset cannot be found
        """
        try:
            csvpath = os.path.join(os.getcwd(), csvfilename)
            rawdf = pd.read_csv(csvpath, encoding="utf8")
            rawdf.drop(columns=["teacher_student_relationship", "stress_level"], axis=1, inplace=True)  # see readme on why this is removed
            rawdf.dropna(inplace=True)  # the provided csv has no null values but imported files may have it
            rawdf.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)  # remove non-ascii chars if applicable
            
            return rawdf
        
        except (TypeError, FileNotFoundError):
            print(
                f"Error when loading dataset at the current working directory in method \"{self._load_dataset.__name__}\".\n"
            +   f"Are you sure your file is in the same directory as this file \"{os.path.basename(__file__)}\"?"
            )
            return None
        
        except pd.errors.ParserError:
            print(
                f"Error when parsing the dataset. Are you sure the dataset is in CSV format?"
            )
            return None
    
    
    def _get_data_points(self, df: pd.DataFrame, label_index=0) -> tuple[list, str]:
        """Retrieves the label and features from the dataset, using
        the first column as the label and the other columns as features.
        Return Format: (Features, Label)
        """
        main_label = df.columns[label_index]
        main_features = df.columns.values.tolist()
        main_features.remove(main_label)  # quick and dirty way to get all other columns
        
        return (main_features, main_label)
    
    
    def view_dataset(self, normalized=False):
        """View the loaded dataset of the class' instance in a terminal.
        Prints the dataset in a terminal-friendly manner.
        This method does not return any values."""
        ds_display = self.raw_ai_dataset
        
        if normalized is True:
            ds_display = self.normalized_ai_dataset
            
        print(ds_display.to_markdown(tablefmt="grid"))
        
        
    def _normalize_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
        """Normalizes the dataset to feed into the AI model
        Returns a normalized dataset as a Pandas DataFrame and the normalizer (MinMaxScaler)
        Return Format: (Normalized DataFrame, MinMaxScaler Object)"""
        normalizer = MinMaxScaler(feature_range=(0, 1))
        
        dfnormarr = normalizer.fit_transform(df)
        
        newdf = pd.DataFrame(dfnormarr, index=df.index, columns=df.columns)
        
        return (newdf, normalizer)
    
    
    def _prepare_dataset(self, df: pd.DataFrame, train_size=0.8, test_size=0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the dataset into features and label DataFrames before
        splitting them into respective datasets for the AI model to interpret.
        Returns four variables of numpy arrays representing the features and labels.
        Return Format: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)"""
        
        featuresdf = df[self.features]
        labelsdf = df[self.label]
        
        trainx, testx, trainy, testy = train_test_split(featuresdf.to_numpy(), labelsdf.to_numpy(), train_size=train_size, test_size=test_size)
        
        return (trainx, testx, trainy, testy)
    
    
    def train_model(self, epochs=30, optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]) -> Sequential:
        """Constructs, trains and validates the AI model on the dataset.
        Keyword arguments are passed into its fit (training) method.
        Returns a compiled model.
        Return Format: Sequential"""
        trainx_shape = self.nn_dataset[0].shape # no need to check if this exists
        print(f"trainx_shape: {trainx_shape}")
        
        model = Sequential([
            Input(shape=trainx_shape[1]), #number of features is input shape
            Dense(64, activation="relu"),
            Dropout(0.2), # prevent model from adapting to training data
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1, activation="linear") # potentially 1 for every number in gad-7 scale
        ])
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        trainx, testx, trainy, testy = self.nn_dataset
        model.fit(trainx, trainy, epochs=epochs)
        
        model.evaluate(testx, testy, verbose=2)
        return model
    
    
    def export_model(self, model: Sequential, name=model_name, filetype=model_suffix) -> Optional[bool]:
        """Exports the AI model (and its weights) into a file with options for customized filetype and filename."""
        try:
            model_directory = os.path.join(os.getcwd(), "static/model/")
            model_filename = name + filetype
            full_model_path = os.path.join(model_directory, model_filename)
            
            model.save(full_model_path)
        
            return True

        except Exception as e:
            print(e)
            return False
    
    
    def load_model(self, file_path=model_directory+model_name+model_suffix) -> Optional[Sequential]:
        """Load an existing AI model from a path to the file.
        Return Format: Model (Sequential) or None"""
        
        if os.path.exists(file_path):
            model = tf.keras.models.load_model(file_path)
            
            return model
            
        else:
            return None # file/directory does not exist
    
    
    def _find_gad_category(self, gad_number: float) -> Optional[str]:
        """Takes a float GAD-7 scaled number and returns an appropriate category if applicable.
        Returns None if the GAD number does not match any defined category ranges.
        Return Format: GAD-7 Category"""
        gad_scaling = [
            ("minimal", (0, 4)),
            ("mild", (5, 9)),
            ("moderate", (10, 14)),
            ("severe", (15, 21))
        ]
            
        gad_names = {category: list(range(start, end+1)) for category, (start, end) in gad_scaling}

        for category, gad_range in gad_names.items():
            if gad_range[0] <= int(gad_number) <= gad_range[-1]: # prevent issues with scores in-between transition
                return category
        return None
        
    
    def predict_anxiety(self, model: Sequential, raw_anx_factors: dict[str, int]) -> Optional[tuple[float, float]]:
        """Use the AI model to predict an anxiety level based on leading factors in the dataset.
        This method does NOT accept a label for AI prediction (defeats the entire purpose).
        Method takes **kwargs for each parameter, with keys being the factors and items being
        the values in their respective non-formatted scaling.
        Return Format: (Normalized Guess, Scaled (GAD-7) Guess, GAD-7 Categorization) or None"""
        try:
            # scikit minmaxscaler requires preserved order to normalize
            ordered_keys = [
                "self_esteem",
                "mental_health_history",
                "depression",
                "headache",
                "blood_pressure",
                "sleep_quality",
                "breathing_problem",
                "noise_level",
                "living_conditions",
                "safety",
                "basic_needs",
                "academic_performance",
                "study_load",
                "future_career_concerns",
                "social_support",
                "peer_pressure",
                "extracurricular_activities",
                "bullying"
            ]
                    
            #anx_factors = {k: [int(v)] for k, v in raw_anx_factors.items()}
            anx_factors = {k: [int(raw_anx_factors[k])] for k in ordered_keys}
            
            rawdf = pd.DataFrame.from_dict(anx_factors)
            rawdf.insert(0, self.label, 0.0, True)  # to allow normalization transform, use float to insert pred as float later
            
            normalized_arr = self.normalizer.transform(rawdf)
            normalized_arr = np.delete(normalized_arr, 0)  # model cannot predict with label column @ 0
            normalized_arr = np.expand_dims(normalized_arr, axis=0)  # add batch size of 1 to shape
            
            prediction = model.predict(normalized_arr)
            
            # normalize data and convert to DataFrame
            normalized_arr = np.insert(normalized_arr, 0, prediction[0][0], axis=-1) # add back into anxiety_level
            
            # inverse_transforms requires that we add back all our existing features/label before giving us the original data            
            resultarr = self.normalizer.inverse_transform(normalized_arr) #normalizeddf
            resultdf = pd.DataFrame(resultarr, index=rawdf.index, columns=rawdf.columns) #, index=normalizeddf.index, columns=normalizeddf.columns
            
            anxiety_raw = prediction[0][0]
            anxiety_scaled = float(resultdf.loc[0, self.label])
            
            gad_score = self._find_gad_category(anxiety_scaled)
            
            return (anxiety_raw, anxiety_scaled, gad_score)
            
        except (ValueError, TypeError) as e:
            print(e)
            return None


if __name__ == "__main__":
    print("MAKING ANXIETY PREDICTOR CLASS")
    instance = AnxietyPredictor()
    
    print("BUILDING MODEL")
    model = instance.train_model()
    
    # You can also import an existing model file instead of building and training
    # model = instance.load_model()
    
    #TEST INFO    
    info = {
        "sleep_quality": 2,
        "noise_level": 2,
        "living_conditions": 2,
        "safety": 2,
        "basic_needs": 2,
        "academic_performance": 2,
        "study_load": 2,
        "future_career_concerns": 2,
        "social_support": 2,
        "headache": 2,
        "blood_pressure": 2,
        "breathing_problem": 2,
        "self_esteem": 15,
        "mental_health_history": 0,
        "depression": 13,
        "peer_pressure": 2,
        "extracurricular_activities": 2,
        "bullying": 2
    }

    
    print("PREDICTING ANXIETY")
    raw_result, scaled_result = instance.predict_anxiety(model, info)
    
    # GAD-7 scores range from 0-21
    print(f"Raw Guess: {raw_result}, GAD-7 Scaled Guess: {scaled_result:2f}")
    
    # Export the model
    # instance.export_model(model)