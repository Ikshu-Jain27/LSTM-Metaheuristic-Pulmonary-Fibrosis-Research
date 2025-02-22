from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def laplace_log_likelihood(y_true, y_pred):
    sigma_clipped = np.maximum(y_pred, 70)
    delta = np.abs(y_true - y_pred)
    return -np.mean(delta / sigma_clipped + np.log(2 * sigma_clipped))

def evaluate_model(model, X_test_text, X_test_img, y_test):
    y_pred = model.predict({'text_input': X_test_text, 'image_input': X_test_img})
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    laplace = laplace_log_likelihood(y_test, y_pred)
    return mse, r2, laplace


# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# def laplace_log_likelihood(y_true, y_pred, sigma=70):
#     sigma_clipped = np.maximum(sigma, 70)
#     delta = np.abs(y_true - y_pred)
#     return -np.mean(delta / sigma_clipped + np.log(2 * sigma_clipped))

# def evaluate_model(model, X_test_text, X_test_img, y_test):
#     y_pred = model.predict([X_test_text, X_test_img]).flatten()
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     laplace = laplace_log_likelihood(y_test, y_pred)
#     return mse, r2, laplace

