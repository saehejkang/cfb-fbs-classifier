import pandas as pd

stat_columns = ["Total Offense", "Rushing Offense", "Passing Offense",
                "Team Passing Efficiency", "Scoring Offense", "Total Defense", "Rushing Defense",
                "Passing Yards Allowed", "Team Passing Efficiency Defense", "Scoring Defense",
                "Turnover Margin", "3rd Down Conversion Pct", "4th Down Conversion Pct",
                "3rd Down Conversion Pct Defense",
                "4th Down Conversion Pct Defense", "Red Zone Offense", "Red Zone Defense", "Net Punting",
                "Punt Returns",
                "Kickoff Returns", "First Downs Offense", "First Downs Defense", "Fewest Penalties Per Game",
                "Fewest Penalty Yards Per Game", "Time of Possession"]


def clean_data() -> pd.DataFrame:
    # read in data from CSV and save as a dataframe
    df = pd.read_csv('../data/cfb_teams_to_id.csv')

    # remove entries where we are missing stats
    # (3 teams end up removed - Jacksonville State, James Madison, Sam Houston)
    df_cleaned = df.dropna(subset=stat_columns)

    # Reset index after dropping rows
    df_cleaned.reset_index(drop=True, inplace=True)

    # go through the time of possession fields and convert the strings to floats
    for x in range(len(df_cleaned)):
        time_val = df_cleaned['Time of Possession'][x]  # get old value
        time_elements = time_val.split(":")
        mins = int(time_elements[0])
        secs = int(time_elements[1])

        # build new value as float (i.e. 30:30 == 30.5, 20:45 == 20.75, 45:00 == 45, etc.)
        time_as_float = float(mins + round(secs / 60, 2))

        # replace the float in the data frame (cleaned)
        df_cleaned.loc[x, 'Time of Possession'] = time_as_float

    return df_cleaned
