from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier, Fourier
import pandas as pd
import pickle as pk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from helpers.transfomers import make_multistep_target
from helpers import aggregate_hourly_data
from helpers.transfomers import make_lags_transformer
from helpers.transfomers import make_leads_transformer
from helpers.date_column_transfomers import is_weekend, get_day_of_week, get_month
from helpers.general_functions import round_to_nearest_n
from sklearn.preprocessing import StandardScaler

APPROX_MULTIYEAR_CYCLE_PERIOD = 365.25*4

def feature_generator(X_input, gate_resolution_level, days_to_predict, target_variable):
    """Generates the features for the model"""

    X = X_input.copy(deep=True)

    # Add columns relating to date variables
    X["DayOfWeek"] = X.apply(get_day_of_week, axis=1)
    X["Month"] = X.apply(get_month, axis=1)
    X["IsWeekend"] = X.apply(is_weekend, axis=1)

    # Calculate an average gate ordinal variable
    X["AverageGateOrdinal"] = round_to_nearest_n(X["AverageGate"], gate_resolution_level)

    # Preprocessing
    # Seasonal features. One for La nina/el nino and another for the annual cycle
    calendar_fourier = CalendarFourier(freq="A", order=1)
    fourier = Fourier(period=APPROX_MULTIYEAR_CYCLE_PERIOD, order=1)

    # Features creation
    dp = DeterministicProcess(
        index=X.index,  # dates from the training data
        constant=False,       # dummy feature for the bias (y_intercept)
        order=0,             # the time dummy (trend)
        additional_terms = [calendar_fourier, fourier],
        drop=True,           # drop terms if necessary to avoid collinearity
    )

    X_seasonal_indicators = dp.in_sample()

    # Days to look back
    n_target_lags = 1
    n_rainfall_lags = 1 
    n_lake_level_lags = 1
    n_rainfall_leads = 3 

    # Bundle preprocessing for data.
    lag_generator = ColumnTransformer(
        transformers=[
            ("target_lags", make_lags_transformer(n_target_lags), [target_variable]),
            #("rainfall_lags", make_lags_transformer(n_rainfall_lags), ["Rainfall"]),
            #("lakelevel_lags", make_lags_transformer(n_lake_level_lags), ["LakeLevel"]),
            ("rainfall_leads", make_leads_transformer(n_rainfall_leads), ["Rainfall"]),
        ],
        remainder='passthrough'
    )

    lead_lag_columns = [
        "Target_lag",
        #"Rainfall_lag",
        #"LakeLevel_lag",
        "Rainfall_lead_1",
        "Rainfall_lead_2",
        "Rainfall_lead_3",
        ]


    # Select data for model fitting
    X_cols_of_interest = X[["Rainfall", target_variable]]

    # Create lags
    X_lags = pd.DataFrame(lag_generator.fit_transform(X_cols_of_interest),
                        index = X.index,
                        columns=lead_lag_columns)
    
    X_features = pd.concat([X_lags, X[["Rainfall", "LakeLevel", "IsWeekend"]]], axis=1)
    
    # Drop nan
    X_features = X_features.dropna()

    # Preprocessing numerical variables
    standardiser_pipeline = Pipeline(steps=[
        #('scale', StandardScaler()),
        #('pca', PCA()),
        ('scaler', StandardScaler())
    ])

    # Bundle preprocessing for all data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', standardiser_pipeline, lead_lag_columns)
        ], 
        remainder = 'passthrough')

    X_preprocessed = pd.DataFrame(data=preprocessor.fit_transform(X_features), columns = X_features.columns, index=X_features.index)

    # Export trained preprocessor
    pk.dump(preprocessor, open("preprocessing/preprocessor.pkl","wb"))

    return X_preprocessed