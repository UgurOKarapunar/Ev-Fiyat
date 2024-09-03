import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Veri setini yükleme ve hazırlama
# Burada df'yi gerçek veri çerçevesi ile değiştirin
# df = pd.read_csv('your_data.csv')  # Örneğin, CSV dosyasından veri yükleme
X = df.drop("Fiyat", axis=1)
y = df["Fiyat"]

# Kategorik ve sayısal özellikleri ayırma
categorical_c = X.select_dtypes(include="object").columns
numerical_c = X.select_dtypes(include=["float", "int"]).columns

# Ön işleme adımları
numerical_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numerical_trans, numerical_c),
        ('cat', categorical_transformer, categorical_c)
    ])

# Veriyi dönüştürme
X_processed = preprocess.fit_transform(X)

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=4)

# Derin öğrenme modeli
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Girdi boyutu
    Dense(64, activation='relu'),          # İlk gizli katman
    Dense(32, activation='relu'),          # İkinci gizli katman
    Dense(1, activation='linear')          # Çıktı katmanı
])

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Model ile tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# Model ve preprocess nesnelerini kaydetme
joblib.dump(preprocess, 'preprocess.pkl')
model.save('model.h5')  # Keras modelini kaydetmek için .save() kullanılır


from matplotlib import pyplot as plt

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.show()
