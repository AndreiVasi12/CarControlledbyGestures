from sklearn.ensemble import RandomForestClassifier
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# Incarcarea datelor
class_names = ['hand_closed', 'one', 'two', 'palm', 'hand_open']
loaded_data = np.load(fr'data\data_{class_names[0]}.npz')
X, y = loaded_data['X'], loaded_data['y']

# Inncărcarea datelor pentru fiecare clasă și combinarea lor intr o singură matrice
for data_class in class_names[1:]:
    loaded_data = np.load(fr'data\data_{data_class}.npz')
    X = np.row_stack((X, loaded_data['X']))
    y = np.concatenate((y, loaded_data['y']))

# Împărțirea datelor intr un set de date de trainig si unul de test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=1, stratify=y)
print(X_train.shape)
print(X_test.shape)

# Antrenarea modelului RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Calcularea acurateței
yhat_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, yhat_train)
print(f'Acuratețea pe setul de antrenament: {train_accuracy}')

# Calcularea acurateței pe setul de testare
yhat_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, yhat_test)
print(f'Acuratețea pe setul de testare: {test_accuracy}')

# Afișarea graficelor
plt.figure(figsize=(10, 6))
plt.hist([y_train, y_test], bins=len(class_names), label=['Train', 'Test'], color=['blue', 'green'])
plt.xlabel('Clase')
plt.ylabel('Frecvență')
plt.title('Distribuția Claselor în Seturile de Antrenament/Testare')
plt.legend()
plt.show()

# Salvarea modelului antrenat
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_str = dt.datetime.strftime(current_date_time_dt, date_time_format)
model_name = f'random_forest_dt_{current_date_time_str}__acc_{test_accuracy}.pkl'
joblib.dump(model, model_name)
