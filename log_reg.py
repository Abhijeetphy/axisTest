import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statsmodels.api import Logit, add_constant

# Load the data (using previously created data as an example)
data_path = 'credit_card_data.csv'
df = pd.read_csv(data_path)
print("dataframe :", df.head())
# Adding a dummy target variable for classification
df['Target'] = np.random.choice([0, 1], size=len(df))

# Step 1: Auto binning with 5 bins and WoE calculation
def calculate_woe_iv(df, feature, target):
    bins = pd.qcut(df[feature], q=5, duplicates='drop')
    grouped = df.groupby(bins)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()
    grouped['WoE'] = np.log(grouped['event_rate'] / grouped['non_event_rate']).replace({np.inf: 0, -np.inf: 0})
    grouped['IV'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['WoE']
    return grouped['WoE'], grouped['IV'].sum()

# Store WoE and IV values
iv_values = {}
woe_transformed_data = {}

target = 'Target'
features = [col for col in df.columns if col not in ['Credit_Card_Number', 'Target']]

for feature in features:
    woe, iv = calculate_woe_iv(df, feature, target)
    iv_values[feature] = iv
    woe_transformed_data[feature] = pd.qcut(df[feature], q=5, duplicates='drop').map(woe)

# Step 2: Remove features with IV below threshold
iv_threshold = 0
selected_features = [feature for feature, iv in iv_values.items() if iv >= iv_threshold]

# Transform the dataset with selected features
woe_df = pd.DataFrame({feature: woe_transformed_data[feature] for feature in selected_features})
woe_df[target] = df[target]
# Step 3: Train a Logistic Regression Model
X = woe_df.drop(columns=[target])
y = woe_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Step 4: Logistic Regression Summary
X_train_const = add_constant(X_train)
model = Logit(y_train, X_train_const).fit()

# Model Summary
print(model.summary())

# Evaluation
y_pred = log_reg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

