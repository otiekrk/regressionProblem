import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.utils import plot_model

insurance_data = pd.read_csv('insurance.csv')

ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

X = insurance_data.drop("charges", axis=1)
y = insurance_data["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

tf.random.set_seed(42)

insurance_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=['mae'])

insurance_model.fit(X_train_normal, y_train, epochs=200)
insurance_model.evaluate(X_test_normal, y_test)

y_pred = insurance_model.predict(X_test_normal)

plot_model(model=insurance_model, show_shapes=True, to_file='model.png', show_layer_names=True)

insurance_model.summary()

indexes = np.arange(0, len(y_test))
plt.figure(figsize=(8, 6))

plt.scatter(indexes, y_test, c="green", label="Testing data")
plt.scatter(indexes, y_pred, c="red", label="Predict data")
plt.legend()
plt.savefig("predictions.png")
plt.show()


insurance_model.save('model.keras')
