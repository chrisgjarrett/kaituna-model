import pandas as pd

def is_weekend(row):
    return (row['DayOfWeek'] == 5) or (row["DayOfWeek"] == 6)

def aggregate_to_daily(hourly_data):
    # Extract hour, day, month, year from TimeStamp
    hourly_data["Hour"] = hourly_data.index.hour
    hourly_data["DayOfWeek"] = hourly_data.index.day_of_week
    hourly_data["DayOfYear"] = hourly_data.index.dayofyear
    hourly_data["Month"] = hourly_data.index.month
    hourly_data["Year"] = hourly_data.index.year

    X_daily = hourly_data.groupby(by=hourly_data.index.date).agg(
        Rainfall=('Rainfall', 'sum'),
        LakeLevel=('LakeLevel', 'mean'),
        DayOfWeek=('DayOfWeek', lambda x: x[0]),
        Month=('Month', lambda x: x[0]),
        #AverageGateOrdinal=('AverageGateOrdinal', lambda x: pd.Series.mode(x)[0]),
        FlowRate=('FlowRate', 'median'),
    )

    X_daily["IsWeekend"] = X_daily.apply(is_weekend, axis=1)

    return X_daily