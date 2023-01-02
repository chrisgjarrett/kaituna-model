from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier, Fourier
import pandas as pd
import pickle as pk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from helpers.transfomers import make_leads_transformer
#from preprocessing.column_transfomer_collection import is_weekend, get_day_of_week, get_month
#from helpers.general_functions import round_to_nearest_n
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

APPROX_MULTIYEAR_CYCLE_PERIOD = 365.25*4

def feature_generator(X_input, target_variable):
    """Generates the features for the model"""

    # Preprocessing
    # Seasonal features. 'fourier' for La nina/el nino and 'calendar_fourier' for the annual cycle
    calendar_fourier = CalendarFourier(freq="A", order=1)
    fourier = Fourier(period=APPROX_MULTIYEAR_CYCLE_PERIOD, order=1)

    # Features creation
    dp = DeterministicProcess(
       index=X_input.index,  # dates from the training data
       constant=False,       # dummy feature for the bias (y_intercept)
       order=0,             # the time dummy (trend)
       additional_terms = [fourier],
       drop=True,           # drop terms if necessary to avoid collinearity
    )

    X_seasonal_indicators = dp.in_sample()

    # Todo - add this to a config file to share with predictor
    features_of_interest = [
        #"Rainfall_lead_1",
        #"Rainfall_lead_2",
        #"Rainfall_lead_3",
        "Rainfall",
        "LakeLevel",
        target_variable,
        ]
    
    # Add seasonal feature columns    
    X_input = X_input[features_of_interest]
    X_input = pd.concat([X_input, X_seasonal_indicators], axis=1)
    X = X_input.copy(deep=True)

    # Preprocessing numerical variables
    pca_pipeline = Pipeline(steps=[
        ('scale', MinMaxScaler()),
        ('pca', PCA()),
        #('scaler', MinMaxScaler())
    ])

    # Preprocessing numerical variables
    standardiser_pipeline = Pipeline(steps=[
        ('scale', MinMaxScaler()),
    ])

    # Bundle preprocessing for all data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', pca_pipeline, ["Rainfall", "LakeLevel"]),
            ('standardiser', standardiser_pipeline, features_of_interest)
        ], 
        remainder = 'passthrough')

    # Fit the preprocessor
    X_preprocessed = pd.DataFrame(data=preprocessor.fit_transform(X), index=X.index, columns=["PCA1", "PCA2", "Rainfall", "LakeLevel", "AverageGate", "Seasonal1","Seasonal2"])

    # Export trained preprocessor
    with open("preprocessing/preprocessor.pkl","wb") as f:
        pk.dump(preprocessor, f)

    return X_preprocessed