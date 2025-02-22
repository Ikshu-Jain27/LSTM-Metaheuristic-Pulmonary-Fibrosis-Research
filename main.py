from train import final_model
from evaluate import evaluate_model
from data_preprocessing import preprocess_text_data, preprocess_image_data
from sklearn.model_selection import train_test_split

# Preprocess data
df, scaler = preprocess_text_data()
patient_ids = df['Patient'].unique()
X_text = df[['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']].values.reshape(-1, 6, 1)
X_img = preprocess_image_data(patient_ids)
y = df['FVC'].values

# Split data
X_text_train, X_text_test, X_img_train, X_img_test, y_train, y_test = train_test_split(X_text, X_img, y, test_size=0.2, random_state=42)

# Evaluate model
mse, r2, laplace = evaluate_model(final_model, X_text_test, X_img_test, y_test)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"Laplace Log Likelihood: {laplace}")

# from train import final_model
# from evaluate import evaluate_model
# from data_preprocessing import preprocess_text_data, preprocess_image_data
# from sklearn.model_selection import train_test_split

# # Preprocess data
# df, scaler = preprocess_text_data()
# patient_ids = df['Patient'].unique()
# X_text = df[['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']].values.reshape(-1, 6, 1)
# X_img = preprocess_image_data(patient_ids)
# y = df['FVC'].values

# # Split data
# X_text_train, X_text_test, X_img_train, X_img_test, y_train, y_test = train_test_split(X_text, X_img, y, test_size=0.2, random_state=42)

# # Evaluate model
# mse, r2, laplace = evaluate_model(final_model, X_text_test, X_img_test, y_test)

# print(f"Mean Squared Error: {mse}")
# print(f"R² Score: {r2}")
# print(f"Laplace Log Likelihood: {laplace}")
