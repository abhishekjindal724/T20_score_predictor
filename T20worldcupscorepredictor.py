import os
import pandas as pd
from tqdm import tqdm
from yaml import safe_load
import numpy as np
import pickle

# Set the local directory where the YAML files are stored
desired_path = "C:\Users\Dell\T20_score_predictor\t20s"  # Update this path
os.chdir(desired_path)

# List all files in the directory and filter out YAML files
directory_contents = os.listdir(desired_path)
yaml_files = [file for file in directory_contents if file.endswith('.yaml') or file.endswith('.yml')]

print(f"Number of YAML files to process: {len(yaml_files)}")

# Prepare an empty list to store all DataFrames
all_dataframes = []
counter = 1

# Loop over each YAML file and process it
for file in tqdm(yaml_files):
    full_path = os.path.join(desired_path, file)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            # Parse the YAML and convert it into a normalized DataFrame
            df = pd.json_normalize(safe_load(f))
            df['match_id'] = counter  # Assign a unique match_id
            all_dataframes.append(df)
            counter += 1
    else:
        print(f"File not found: {full_path}")

# Concatenate all individual DataFrames into one large DataFrame
final_df = pd.concat(all_dataframes)

# Print the combined DataFrame
print(final_df)

# Backup the final DataFrame
backup_df = final_df.copy()

# Display the DataFrame (optional in local environments)
print(final_df)

# Drop unnecessary columns
final_df.drop(columns=[
    'meta.data_version',
    'meta.created',
    'meta.revision',
    'info.outcome.bowl_out',
    'info.bowl_out',
    'info.supersubs.South Africa',
    'info.supersubs.New Zealand',
    'info.outcome.eliminator',
    'info.outcome.result',
    'info.outcome.method',
    'info.neutral_venue',
    'info.match_type_number',
    'info.outcome.by.runs',
    'info.outcome.by.wickets'
], inplace=True)

# Filter only male matches and 20-over matches
final_df = final_df[final_df['info.gender'] == 'male']
final_df.drop(columns=['info.gender'], inplace=True)

final_df = final_df[final_df['info.overs'] == 20]
final_df.drop(columns=['info.overs', 'info.match_type'], inplace=True)

# Save the processed DataFrame locally
pickle.dump(final_df, open('dataset_level1.pkl', 'wb'))

# Load the saved data (for verification)
matches = pickle.load(open('dataset_level1.pkl', 'rb'))

# Process the deliveries for each match
count = 1
delivery_df = pd.DataFrame()

for index, row in matches.iterrows():
    count += 1
    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []
    match_id = []
    city = []
    venue = []
    
    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            match_id.append(count)
            batting_team.append(row['innings'][0]['1st innings']['team'])
            teams.append(row['info.teams'])
            ball_of_match.append(key)
            batsman.append(ball[key]['batsman'])
            bowler.append(ball[key]['bowler'])
            runs.append(ball[key]['runs']['total'])
            city.append(row['info.city'])
            venue.append(row['info.venue'])
            
            # Handle dismissals
            try:
                player_of_dismissed.append(ball[key]['wicket']['player_out'])
            except:
                player_of_dismissed.append('0')

    # Create DataFrame for each match and concatenate
    loop_df = pd.DataFrame({
        'match_id': match_id,
        'teams': teams,
        'batting_team': batting_team,
        'ball': ball_of_match,
        'batsman': batsman,
        'bowler': bowler,
        'runs': runs,
        'player_dismissed': player_of_dismissed,
        'city': city,
        'venue': venue
    })

    delivery_df = pd.concat([delivery_df, loop_df], ignore_index=True)

# Function to determine the bowling team
def bowl(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team

# Apply function to get the bowling team
delivery_df['bowling_team'] = delivery_df.apply(bowl, axis=1)

# Drop unnecessary columns
delivery_df.drop(columns=['teams'], inplace=True)

# Filter for specific teams
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand',
    'South Africa', 'England', 'West Indies', 
    'Afghanistan', 'Pakistan', 'Sri Lanka'
]

delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]

# Select relevant columns for the final output
output = delivery_df[['match_id', 'batting_team', 'bowling_team', 'ball', 'runs', 'player_dismissed', 'city', 'venue']]

# Save the processed deliveries data
pickle.dump(output, open('dataset_level2.pkl', 'wb'))

# Load the data (for verification)
df = pickle.load(open('dataset_level2.pkl', 'rb'))
print(df)
