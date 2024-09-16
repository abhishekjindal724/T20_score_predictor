<<<<<<< HEAD
import pandas as pd
import pickle
import numpy as np

# Load the dataset (stored locally instead of Colab)
df = pickle.load(open('dataset_level2.pkl','rb'))

# Check for null values
print(df.isnull().sum())

# Replace missing cities with the first part of the venue name
print(df[df['city'].isnull()]['venue'].value_counts())

# Handle missing cities by using the venue name
cities = np.where(df['city'].isnull(), df['venue'].str.split().apply(lambda x: x[0]), df['city'])

df['city'] = cities

# Re-check for missing values
print(df.isnull().sum())

# Drop the 'venue' column
df.drop(columns=['venue'], inplace=True)

# Filter eligible cities (those with more than 600 occurrences)
eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 600].index.tolist()
df = df[df['city'].isin(eligible_cities)]

# Calculate the current score by match
df['current_score'] = df.groupby('match_id')['runs'].cumsum()

# Extract over and ball number
df['over'] = df['ball'].apply(lambda x: str(x).split(".")[0])
df['ball_no'] = df['ball'].apply(lambda x: str(x).split(".")[1])

# Calculate balls bowled and balls left
df['balls_bowled'] = (df['over'].astype('int') * 6) + df['ball_no'].astype('int')
df['balls_left'] = 120 - df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x: 0 if x < 0 else x)

# Handle player dismissals
df['player_dismissed'] = df['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
df['player_dismissed'] = df['player_dismissed'].astype('int')
df['player_dismissed'] = df.groupby('match_id')['player_dismissed'].cumsum()

# Calculate wickets left
df['wickets_left'] = 10 - df['player_dismissed']

# Calculate the current run rate (crr)
df['crr'] = (df['current_score'] * 6) / df['balls_bowled']

# Group by match and calculate the last five overs runs
groups = df.groupby('match_id')
match_ids = df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=30)['runs'].sum().values.tolist())

df['last_five'] = last_five

# Create the final dataset
final_df = df.groupby('match_id')['runs'].sum().reset_index().merge(df, on='match_id')
final_df = final_df[['batting_team', 'bowling_team', 'city', 'current_score', 'balls_left', 'wickets_left', 'crr', 'last_five', 'runs_x']]

# Drop any remaining NaN values and shuffle the dataset
final_df.dropna(inplace=True)
final_df = final_df.sample(final_df.shape[0])

# Split into features (X) and target (y)
X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define the preprocessing and model pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Define the ColumnTransformer for categorical variables
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Define the full pipeline with the model
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1))
])

# Train the model
pipe.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipe.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

# Save the trained model locally
with open('pipe.pkl', 'wb') as model_file:
    pickle.dump(pipe, model_file)

print("Model saved as 'pipe.pkl'")
=======
import pandas as pd
import pickle
import numpy as np

# Load the dataset (stored locally instead of Colab)
df = pickle.load(open('dataset_level2.pkl','rb'))

# Check for null values
print(df.isnull().sum())

# Replace missing cities with the first part of the venue name
print(df[df['city'].isnull()]['venue'].value_counts())

# Handle missing cities by using the venue name
cities = np.where(df['city'].isnull(), df['venue'].str.split().apply(lambda x: x[0]), df['city'])

df['city'] = cities

# Re-check for missing values
print(df.isnull().sum())

# Drop the 'venue' column
df.drop(columns=['venue'], inplace=True)

# Filter eligible cities (those with more than 600 occurrences)
eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 600].index.tolist()
df = df[df['city'].isin(eligible_cities)]

# Calculate the current score by match
df['current_score'] = df.groupby('match_id')['runs'].cumsum()

# Extract over and ball number
df['over'] = df['ball'].apply(lambda x: str(x).split(".")[0])
df['ball_no'] = df['ball'].apply(lambda x: str(x).split(".")[1])

# Calculate balls bowled and balls left
df['balls_bowled'] = (df['over'].astype('int') * 6) + df['ball_no'].astype('int')
df['balls_left'] = 120 - df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x: 0 if x < 0 else x)

# Handle player dismissals
df['player_dismissed'] = df['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
df['player_dismissed'] = df['player_dismissed'].astype('int')
df['player_dismissed'] = df.groupby('match_id')['player_dismissed'].cumsum()

# Calculate wickets left
df['wickets_left'] = 10 - df['player_dismissed']

# Calculate the current run rate (crr)
df['crr'] = (df['current_score'] * 6) / df['balls_bowled']

# Group by match and calculate the last five overs runs
groups = df.groupby('match_id')
match_ids = df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=30)['runs'].sum().values.tolist())

df['last_five'] = last_five

# Create the final dataset
final_df = df.groupby('match_id')['runs'].sum().reset_index().merge(df, on='match_id')
final_df = final_df[['batting_team', 'bowling_team', 'city', 'current_score', 'balls_left', 'wickets_left', 'crr', 'last_five', 'runs_x']]

# Drop any remaining NaN values and shuffle the dataset
final_df.dropna(inplace=True)
final_df = final_df.sample(final_df.shape[0])

# Split into features (X) and target (y)
X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define the preprocessing and model pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Define the ColumnTransformer for categorical variables
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Define the full pipeline with the model
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1))
])

# Train the model
pipe.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipe.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

# Save the trained model locally
with open('pipe.pkl', 'wb') as model_file:
    pickle.dump(pipe, model_file)

print("Model saved as 'pipe.pkl'")
>>>>>>> c44a98faf6e9c99b83c56c9caf77df8732f2bf28
