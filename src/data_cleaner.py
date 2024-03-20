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
    #read in data from CSV and save as a dataframe
    df = pd.read_csv('../data/cfb_teams_to_id.csv')

    #remove entries where we are missing stats (3 teams end up removed - Jacksonville State, James Madison, Sam Houston)
    df_cleaned = df.dropna(subset=stat_columns)

    return df_cleaned

    # # isolate the features into their own Dataframe
    # x = df.iloc[:, 3:]
    # y = df['Final_Standing']
    #
    # print(X)
    # print(y)
