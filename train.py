from pyswarm import pso
from model import create_fusion_model
from data_preprocessing import preprocess_text_data, preprocess_image_data

df, _ = preprocess_text_data()
X_text = df[['Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']].values.reshape(-1, 6, 1)
X_img = preprocess_image_data(df['Patient'].unique(), df)  # Pass full df to match lengths
y = df['FVC'].values


def objective_function(params):
    units, dropout = int(params[0]), params[1]

    model = create_fusion_model()
    model.fit({'text_input': X_text, 'image_input': X_img}, y, epochs=3, batch_size=32, verbose=0)
    loss = model.evaluate({'text_input': X_text, 'image_input': X_img}, y, verbose=0)[0]  # Extract loss



    return float(loss)

# PSO Optimization
best_params, _ = pso(objective_function, [32, 0.1], [128, 0.5])

# Train final model with optimized parameters
final_model = create_fusion_model()
final_model.fit([X_text, X_img], y, epochs=10, batch_size=32, verbose=1)
final_model.save('fusion_model.h5')


# from pyswarm import pso
# from model import create_fusion_model
# from data_preprocessing import preprocess_text_data, preprocess_image_data
# from sklearn.model_selection import train_test_split

# def objective_function(params):
#     units, dropout = int(params[0]), params[1]
#     model = create_fusion_model(units, dropout)
#     df, scaler = preprocess_text_data()
#     patient_ids = df['Patient'].unique()
#     X_text, X_img, y = preprocess_image_data(patient_ids, df)
#     X_text_train, _, X_img_train, _, y_train, _ = train_test_split(X_text, X_img, y, test_size=0.2, random_state=42)
#     model.fit([X_text_train, X_img_train], y_train, epochs=5, batch_size=32, verbose=0)
#     loss = model.evaluate([X_text_train, X_img_train], y_train, verbose=0)
#     return loss

# best_params, _ = pso(objective_function, [32, 0.1], [128, 0.5])
# final_model = create_fusion_model(int(best_params[0]), best_params[1])
# df, scaler = preprocess_text_data()
# patient_ids = df['Patient'].unique()
# X_text, X_img, y = preprocess_image_data(patient_ids, df)
# final_model.fit([X_text, X_img], y, epochs=50, batch_size=32, verbose=1)
# final_model.save('fusion_model.h5')
