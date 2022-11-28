
def is_weekend(row):    
        return (row['DayOfWeek'] == 5) or (row["DayOfWeek"] == 6)

def get_day_of_week(row):
        return row.name.dayofweek

def get_month(row):
        return row.name.month
