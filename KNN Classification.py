import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Pobranie danych z GitHub
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=column_names)

# Podział danych na zmienne wejściowe i przewidywane
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Podział danych na testowe i uczące
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Zmienne przechowujące najlepszą liczbę sąsiadów oraz dokładność
best_accuracy = 0
best_k = 0

# Pętla dla różnych wartości zmiennej k
for k in range(3, 11, 2):
    # klasyfikator KNN
    knn = KNeighborsClassifier(n_neighbors=k)

    # Dopasowanie klasyfikatora do danych uczących
    knn.fit(X_train, y_train)

    # Predykcja danych testowych
    y_pred_test = knn.predict(X_test)

    # Ocena dopasowania do danych testowych
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Sprawdzenie, czy bieżąca wartość k daje najlepsze dopasowanie
    if accuracy_test > best_accuracy:
        best_accuracy = accuracy_test
        best_k = k

print("Najlepsze dopasowanie:", best_accuracy)
print("Optymalna liczba sąsiadów:", best_k)

# Wyznaczenie macierzy pomyłek
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Wyświetlenie raportu klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred_test, target_names=['No Diabetes', 'Diabetes']))

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Greens", fmt="d",
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek')
plt.show()

# Funkcja do wyznaczania granic decyzyjnych
def plot_decision_boundary(X, y, classifier, title):
    h = 0.2  # rozmiar kroku w siatce
    cmap_light = ListedColormap(['#FFB3BA', '#B3E2CD'])
    cmap_bold = ListedColormap(['#FF6666', '#66C2A5'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Zmienna niezależna')
    plt.ylabel('Zmienna objaśniająca')
    plt.show()

# zmniejszenie skali dla wizualizacji
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# Ponowne dopasowanie kalsyfikatora KNN do liczby sąsiadów
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_2d, y_train)

# Wizualizacja granic decyzyjnych dla zbioru uczącego
plot_decision_boundary(X_train_2d, y_train, knn, f'Granice decyzyjne dla KNN (K={best_k}) - Zbiór uczący')

# Wizualizacja granic decyzyjnych dla zbioru testowego
plot_decision_boundary(X_test_2d, y_test, knn, f'Granice decyzyjne dla KNN (K={best_k}) - Zbiór testowy')

