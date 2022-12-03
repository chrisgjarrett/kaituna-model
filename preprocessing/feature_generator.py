from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier, Fourier
import pandas as pd
import pickle as pk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from helpers.transfomers import make_leads_transformer
from preprocessing.column_transfomer_collection import is_weekend, get_day_of_week, get_month
from helpers.general_functions import round_to_nearest_n
from sklearn.preprocessing import StandardScaler

APPROX_MULTIYEAR_CYCLE_PERIOD = 365.25*4

def feature_generator(X_input, target_variable):
    """Generates the features for the model"""

    # Todo - add this to a config file to share with predictor
    features_of_interest = [
        "Rainfall_lead_1",
        "Rainfall_lead_2",
        "Rainfall_lead_3",
        "Rainfall",
        "LakeLevel",
        target_variable
        ]
    #X_input = X_input[features_of_interest]
    X = X_input.copy(deep=True)

    # Preprocessing numerical variables
    standardiser_pipeline = Pipeline(steps=[
        #('scale', StandardScaler()),
        #('pca', PCA()),
        ('scaler', StandardScaler())
    ])

    # Bundle preprocessing for all data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', standardiser_pipeline, features_of_interest)
        ], 
        remainder = 'drop')

    X_preprocessed = pd.DataFrame(data=preprocessor.fit_transform(X), columns = features_of_interest, index=X.index)

    # Export trained preprocessor
    pk.dump(preprocessor, open("preprocessing/preprocessor.pkl","wb"))

    return X_preprocessed


    # Preprocessing
    # Seasonal features. One for La nina/el nino and another for the annual cycle
    #calendar_fourier = CalendarFourier(freq="A", order=1)
    #fourier = Fourier(period=APPROX_MULTIYEAR_CYCLE_PERIOD, order=1)

    # Features creation
    #dp = DeterministicProcess(
    #    index=X.index,  # dates from the training data
    #    constant=False,       # dummy feature for the bias (y_intercept)
    #    order=0,             # the time dummy (trend)
    #    additional_terms = [calendar_fourier, fourier],
    #    drop=True,           # drop terms if necessary to avoid collinearity
    #)

    #X_seasonal_indicators = dp.in_sample()
