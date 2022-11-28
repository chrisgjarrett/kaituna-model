import pandas as pd

GATE_RESOLUTION_LEVEL = 100

def aggregate_hourly_data(hourly_df):
    """Takes a hourly dataset and converts to daily. Assumes the following columns are present:
    - TimeStamp, Gate1, Gate2, Gate3, Rainfall, LakeLevel, FlowRate
    """
    
    # Replace individual gates with average
    hourly_df["AverageGate"] = hourly_df[["Gate1", "Gate2", "Gate3"]].median(axis=1)
    hourly_df=hourly_df.drop(["Gate1","Gate2","Gate3"], axis=1)

    # Aggregate to daily metrics
    daily_data = hourly_df.groupby(by=hourly_df.index.date).agg(
        Rainfall=('Rainfall', 'sum'),
        LakeLevel=('LakeLevel', 'mean'),
        AverageGate=('AverageGate', lambda x: pd.Series.mode(x)[0]),
        FlowRate=('FlowRate', 'median'),
    )

    daily_data.index = pd.to_datetime(daily_data.index)
    daily_data.set_index(daily_data.index.to_period('D'), inplace=True)

    return daily_data

# When run as a module, save the dataframe
if __name__=="__main__":
    hourly_df = pd.read_csv('hourly_kaituna_data.csv', parse_dates=["TimeStamp"])
    daily_data = aggregate_hourly_data(hourly_df)
    # Save to csv
    daily_data.to_csv('kaituna_daily.csv')