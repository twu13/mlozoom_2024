import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from URLs
yellow_01_23 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet")
yellow_02_23 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet")

# 1
print(f"Number of columns: {len(yellow_01_23.columns)}")

# 2
duration = yellow_01_23.tpep_dropoff_datetime - yellow_01_23.tpep_pickup_datetime
duration_min = (duration.dt.total_seconds() / 60).astype(float)
print(f"Standard deviation of trip duration (minutes): {duration_min.std()}")

# 3
initial_recs = len(duration)
final_recs = len(duration_min[(duration_min >= 1) & (duration_min <= 60)])
print(f"Fraction of valid records: {final_recs / initial_recs:.2f}")

# 4
df_train = yellow_01_23.copy()
df_train["duration"] = duration_min
df_train = df_train[(df_train.duration >= 1) & (df_train.duration <= 60)]

# Convert categorical variables to strings and vectorize them
categorical = ['PULocationID', 'DOLocationID']
df_train[categorical] = df_train[categorical].astype(str)
train_dicts = df_train[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
print(f"Number of columns: {X_train.shape[1]}")

# 5
y_train = df_train['duration'].values
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model on the training data
y_pred_train = lr.predict(X_train)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
print(f"Training RMSE: {train_rmse:.2f}")

# 6
df_val = yellow_02_23.copy()
duration_val = df_val.tpep_dropoff_datetime - df_val.tpep_pickup_datetime
duration_val_min = (duration_val.dt.total_seconds() / 60).astype(float)
df_val["duration"] = duration_val_min
df_val = df_val[(df_val.duration >= 1) & (df_val.duration <= 60)]
df_val[categorical] = df_val[categorical].astype(str)

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = df_val['duration'].values

# Predict and evaluate the model on the validation data
y_pred_val = lr.predict(X_val)
val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
print(f"Validation RMSE: {val_rmse:.2f}")