# ML Bible - Clasificación de Modelos

## Descripción

En este repositorio se exploran diversos modelos de clasificación utilizando scikit-learn, evaluando su rendimiento mediante métricas comunes como el `classification_report` y la `cross-validation score`. A continuación, te presento el código principal donde se entrenan varios modelos y se evalúan sus desempeños.

---

## Código de Clasificación

Este script se encarga de entrenar varios modelos y evaluar su desempeño en el conjunto de datos de entrenamiento y prueba.

```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier()
}

# Entrenamiento y evaluación de los modelos
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Modelo: {name}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Evaluación utilizando cross-validation
for name, model in models.items():
    cv_score = cross_val_score(model, X, y, cv=5)
    print(f"{name} - CV Score: {cv_score.mean()}")

    if cv_score.mean() > best_score:
        best_score = cv_score.mean()
        best_model = model
        best_model_name = name

print(f"The best model is: {best_model_name} with a CV mean of {best_score}")

if best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l2']
    }
elif best_model_name == "SVM":
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
elif best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    }
elif best_model_name == "KNN":
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best hyperparameters for {best_model_name}: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")


best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)


print(f"\nConfusion matrix of the best model ({best_model_name}):")
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
```
