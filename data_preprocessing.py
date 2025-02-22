import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from config import DATA_PATH, IMAGE_PATH, IMG_SIZE
import os

def preprocess_text_data():
    df = pd.read_csv(DATA_PATH)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['SmokingStatus'] = LabelEncoder().fit_transform(df['SmokingStatus'])
    
    features = ['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, scaler

def preprocess_image_data(patient_ids, df):
    images = []
    valid_patient_ids = set(os.listdir(IMAGE_PATH))  # Get available images

    for pid in df['Patient'].values:  # Iterate over all patients in df
        img_filename = f'{pid}.png'
        img_path = os.path.join(IMAGE_PATH, img_filename)

        if img_filename in valid_patient_ids and os.path.exists(img_path):
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        else:
            images.append(np.zeros((*IMG_SIZE, 3)))  # Handle missing images

    return np.array(images)


# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from config import DATA_PATH, IMAGE_PATH, IMG_SIZE
# import os

# def preprocess_text_data():
#     df = pd.read_csv(DATA_PATH)
#     df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
#     df['SmokingStatus'] = LabelEncoder().fit_transform(df['SmokingStatus'])
    
#     features = ['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']
#     scaler = MinMaxScaler()
#     df[features] = scaler.fit_transform(df[features])
    
#     return df, scaler

# def preprocess_image_data(patient_ids, df):
#     images, valid_ids = [], []
#     for pid in patient_ids:
#         img_path = os.path.join(IMAGE_PATH, f'{pid}.png')
#         if os.path.exists(img_path):
#             img = load_img(img_path, target_size=IMG_SIZE)
#             img_array = img_to_array(img) / 255.0
#             images.append(img_array)
#             valid_ids.append(pid)
#         else:
#             images.append(np.zeros((*IMG_SIZE, 3)))
#             valid_ids.append(pid)
    
#     df_filtered = df[df['Patient'].isin(valid_ids)]
#     X_text = df_filtered[['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']].values.reshape(-1, 6, 1)
#     y = df_filtered['FVC'].values
#     return np.array(X_text), np.array(images), np.array(y)