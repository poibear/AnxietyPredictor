## TODO optimize predict method via numpy calculations
# Disable Tensorflow Warnings/Errors
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class AnxietyPredictor:
    """An AI that can predict anxiety levels, according to the GAD-7 scale, based
    on leading factors of anxiety."""
    
    default_csv_filename = "stress_level_dataset.csv"
    
    def __init__(self):
        # preparation stage
        self.raw_ai_dataset = self._load_dataset()
        self.features, self.label = self._get_data_points(self.raw_ai_dataset)
        
        # preprocessing stage
        self.normalized_ai_dataset, self.normalizer = self._normalize_dataset(self.raw_ai_dataset)
        self.nn_dataset = self._prepare_dataset(self.normalized_ai_dataset)
    
    
    def _load_dataset(self, csvfilename=default_csv_filename) -> pd.DataFrame:
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
        
        except pd.errors.ParserError:
            print(
                f"Error when parsing the dataset. Are you sure the dataset is in CSV format?"
            )
    
    
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
    
    
    def build_model(self):
        """Constructs and returns an AI (Sequential) model.
        Return Format: Sequential Model"""
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
        
        return model
    
    
    def train_model(self, model: Sequential, epochs=30, optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]) -> Sequential:
        """Trains and validates the AI model on the dataset.
        Keyword arguments are passed into its fit (training) method.
        Returns the trained model.
        Return Format: Sequential"""
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        trainx, testx, trainy, testy = self.nn_dataset
        model.fit(trainx, trainy, epochs=epochs)
        
        return model.evaluate(testx, testy, verbose=2)
    
    
    def predict_anxiety(self, model: Sequential, raw_anx_factors: dict[str, int]) -> tuple[float, float]:
        """Use the AI model to predict an anxiety level based on leading factors in the dataset.
        This method does NOT accept a label for AI prediction (defeats the entire purpose).
        Method takes **kwargs for each parameter, with keys being the factors and items being
        the values in their respective non-formatted scaling.
        Return Format: (Normalized Guess, Scaled (GAD-7) Guess)"""
        try:
            anx_factors = {k: [v] for k, v in raw_anx_factors.items()}
            
            rawdf = pd.DataFrame.from_dict(anx_factors)
            rawdf.insert(0, self.label, 0.0, True)  # to allow normalization transform, use float to insert pred as float later
            
            normalized_arr = self.normalizer.transform(rawdf)
            normalized_arr = np.delete(normalized_arr, 0)  # model cannot predict with label column @ 0
            normalized_arr = np.expand_dims(normalized_arr, axis=0)  # add batch size of 1 to shape
            
            prediction = model.predict(normalized_arr)
            
            # normalize data and convert to DataFrame
            normalized_arr = np.insert(normalized_arr, 0, prediction[0][0], axis=-1) # add back into anxiety_level
            
            # this sucks but inverse_transforms requires that we add back all our existing features/label before giving us the original data            
            resultarr = self.normalizer.inverse_transform(normalized_arr) #normalizeddf
            resultdf = pd.DataFrame(resultarr, index=rawdf.index, columns=rawdf.columns) #, index=normalizeddf.index, columns=normalizeddf.columns
            
            anxiety_raw = prediction[0][0]
            anxiety_scaled = float(resultdf.loc[0, self.label])
            
            return (anxiety_raw, anxiety_scaled)
            
        except ValueError as e:
            print(e)
            print(
                f"Error when adding anxiety factors to a DataFrame in method \"{self.predict_anxiety.__name__}\". Check your keyword arguments.\n"
            +   f"Are you sure that you entered the same number and names of KEYS ({len(anx_factors.keys())}) as the FEATURES ({self.normalized_ai_dataset.shape[1]-1}) in your dataset?"
            )
        
        except TypeError:
            print(
                f"Error when importing arguments to method \"{self.predict_anxiety.__name__}\".\n"
            +   "Are you sure you've define your dictionary as a keyword argument (e.g., raw_anx_factors={\"foo\": 3})?"
            +   "Make sure your (Sequential) model is passed as your first argument for this method (from build_model())."
            )


if __name__ == "__main__":
    print("MAKING ANXIETY PREDICTOR CLASS")
    instance = AnxietyPredictor()
    
    print("BUILDING MODEL")
    model = instance.build_model()
    
    print("TRAINING MODEL")
    instance.train_model(model)
    
    # test information
    info = {
        "self_esteem": 19,
        "mental_health_history": 0,
        "depression": 3,
        "headache": 2,
        "blood_pressure": 1,
        "sleep_quality": 2,
        "breathing_problem": 2,
        "noise_level": 2,
        "living_conditions": 0,
        "safety": 4,
        "basic_needs": 4,
        "academic_performance": 3,
        "study_load": 2,
        "future_career_concerns": 4,
        "social_support": 3,
        "peer_pressure": 2,
        "extracurricular_activities": 1,
        "bullying": 4
    }
    
    print("PREDICTING ANXIETY")
    raw_result, scaled_result = instance.predict_anxiety(model, info)
    # gad-7 ranges from 0-21
    print(f"Raw Guess: {raw_result}, GAD-7 Scaled Guess: {scaled_result:2f}")