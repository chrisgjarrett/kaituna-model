import requests
import pandas as pd

def get_df_from_json_bop_rivers(data_url, record_path=['Data']):
    content = requests.get(data_url).json()
    df_all = pd.json_normalize(content, record_path=record_path)

    return df_all

def collate_kaituna_data(start_date, end_date):
    # Get flow data
    flow_data_url = 'https://envdata.boprc.govt.nz/Data/DatasetGrid?dataset=35946&sort=TimeStamp-desc&page=1&group=&filter=&interval=Custom&timezone=720&date={}&endDate={}&calendar=1&alldata=false'.format(
        start_date, end_date)
    flow_df_all = get_df_from_json_bop_rivers(flow_data_url)
    flow_df = flow_df_all[["TimeStamp", "Value"]]
    flow_df = flow_df.rename(columns={'Value': 'FlowRate'})
    flow_df['TimeStamp'] = pd.to_datetime(flow_df['TimeStamp'])

    # Get Lake levels
    lake_level_url = 'https://envdata.boprc.govt.nz/Data/DatasetGrid?dataset=32419&sort=TimeStamp-desc&page=1&group=&filter=&interval=Custom&timezone=720&date={}&endDate={}&calendar=1&alldata=false'.format(
        start_date, end_date)
    lake_level_df_all = get_df_from_json_bop_rivers(lake_level_url)
    lake_level_df = lake_level_df_all[["TimeStamp", "Value"]]
    lake_level_df = lake_level_df.rename(columns={'Value': 'LakeLevel'})
    lake_level_df['TimeStamp'] = pd.to_datetime(lake_level_df['TimeStamp'])

    # Get gate levels
    # Gate 1
    gate_levels_url_1 = 'https://envdata.boprc.govt.nz/Data/DatasetGrid?dataset=38970&sort=TimeStamp-desc&page=1&group=&filter=&interval=Custom&timezone=720&date={}&endDate={}&calendar=1&alldata=false'.format(
        start_date, end_date)
    gate_1_df = get_df_from_json_bop_rivers(gate_levels_url_1)
    gate_1_df = gate_1_df[["TimeStamp", "Value"]]
    gate_1_df = gate_1_df.rename(columns={"Value": "Gate1"})

    # Gate 2
    gate_levels_url_2 = 'https://envdata.boprc.govt.nz/Data/DatasetGrid?dataset=38973&sort=TimeStamp-desc&page=1&group=&filter=&interval=Custom&timezone=720&date={}&endDate={}&calendar=1&alldata=false'.format(
        start_date, end_date)
    gate_2_df = get_df_from_json_bop_rivers(gate_levels_url_2)
    gate_2_df = gate_2_df[["TimeStamp", "Value"]]
    gate_2_df = gate_2_df.rename(columns={"Value": "Gate2"})

    # Gate 3
    gate_levels_url_3 = 'https://envdata.boprc.govt.nz/Data/DatasetGrid?dataset=38972&sort=TimeStamp-desc&page=1&group=&filter=&interval=Custom&timezone=720&date={}&endDate={}&calendar=1&alldata=false'.format(
        start_date, end_date)
    gate_3_df = get_df_from_json_bop_rivers(gate_levels_url_3)
    gate_3_df = gate_3_df[["TimeStamp", "Value"]]
    gate_3_df = gate_3_df.rename(columns={"Value": "Gate3"})

    # Concatenate into single dataframe
    gate_levels_df_temp = pd.merge(gate_1_df, gate_2_df, on='TimeStamp')
    gate_levels_df = pd.merge(gate_levels_df_temp, gate_3_df, on='TimeStamp', how='left')
    gate_levels_df['TimeStamp'] = pd.to_datetime(gate_levels_df['TimeStamp'])

    # Get rainfall at Lake Rotoiti
    rainfall_url = 'https://envdata.boprc.govt.nz/Data/DatasetGrid?dataset=47928&sort=TimeStamp-desc&page=1&group=&filter=&interval=Custom&timezone=720&date={}&endDate={}&calendar=1&alldata=false'.format(
        start_date, end_date)
    rainfall_df_all = get_df_from_json_bop_rivers(rainfall_url)
    rainfall_df = rainfall_df_all[["TimeStamp", "Value"]]
    rainfall_df = rainfall_df.rename(columns={'Value': 'Rainfall'})
    rainfall_df['TimeStamp'] = pd.to_datetime(rainfall_df['TimeStamp'])

    # Merge datasets into one dataframe
    df_temp = pd.merge(lake_level_df, flow_df, on='TimeStamp', how='left')
    df_temp = pd.merge(df_temp, gate_levels_df, on='TimeStamp', how='left')
    df_combined = pd.merge(df_temp, rainfall_df, on='TimeStamp', how='right')

    # Set index
    df_combined['TimeStamp'] = pd.to_datetime(df_combined['TimeStamp'])
    df_combined.set_index('TimeStamp', inplace=True)

    # Remove rows for which we don't have rainfall data
    df_river_data = df_combined.dropna()

    # Sort by timestamp descending
    df_river_data = df_river_data.sort_values('TimeStamp')

    df_river_data = df_river_data.astype(float)

    return df_river_data