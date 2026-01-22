This project contains a machine learning model that predicts house prices in Melbourne using historical housing data.

The model is trained on a dataset containing features such as:
1. Number of rooms.
2. Distance from the city.
3. Land size.
4. Building area.
5. Year Built.
6. And other relevant features.
It predicts the house price using XGBoost regression.

HOW IT WORKS:
1. Data is loaded from melb_data.csv.
2. Preprocessing is done: Missing values are handled & Categorical values are encoded.
3. Model training using XGBRegressor.
4. Prediction is made on test/validation data.
5. Model performance is evaluated using MAE(Mean Absolute Error)

OUTPUT:
The script will output:
1. Model MAE(Mean Absolute Error).
2. Predicted prices for the validation dataset.
