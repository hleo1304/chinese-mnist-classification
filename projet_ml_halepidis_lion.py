# Projet Machine Learning : Chinese MNIST
# Léo Halepidis & Maxence Lion

import os
# Coupe les warnings TF et oneDNN pour garder les mêmes arrondis
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD

import warnings
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
keras.utils.set_random_seed(seed)

DOSSIER_BASE = os.path.dirname(os.path.abspath(__file__))
DOSSIER_RESULTS = os.path.join(DOSSIER_BASE, "results")

if not os.path.exists(DOSSIER_RESULTS):
    os.makedirs(DOSSIER_RESULTS)

# -------------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------------
print("Chargement des images en cours...")

df = pd.read_csv(os.path.join(DOSSIER_BASE, "chinese_mnist.csv"))

X = []
y = []
dico_labels = {}

for i, ligne in df.iterrows():
    code = int(ligne["code"])
    dico_labels[code] = str(ligne["value"])
    
    chemin_img = os.path.join(DOSSIER_BASE, "data", f"input_{ligne['suite_id']}_{ligne['sample_id']}_{code}.jpg")
    
    img = Image.open(chemin_img).convert("L").resize((32, 32), resample=Image.Resampling.LANCZOS)  # garde les traits nets
    X.append(np.array(img).flatten() / 255.0)
    y.append(code)

X = np.array(X, dtype=np.float32)
y = np.array(y)

codes_tries = sorted(list(dico_labels.keys()))
noms_classes = [dico_labels[c] for c in codes_tries]

print(f"Total : {len(X)} images chargées (taille d'une image aplatie : {X.shape[1]} pixels)")

# -------------------------------------------------------------
# 2. Découpage train / test (80% / 20%)
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed, stratify=y
)

print(f"Train : {len(X_train)} images | Test : {len(X_test)} images")

# 3000 images du train pour faire tourner les GridSearch plus vite
X_train_reduit, _, y_train_reduit, _ = train_test_split(
    X_train, y_train, train_size=3000, random_state=seed, stratify=y_train
)

def tracer_matrice(y_true, y_pred, titre, nom_png):
    cm = confusion_matrix(y_true, y_pred, labels=codes_tries)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=noms_classes, yticklabels=noms_classes)
    plt.title(titre)
    plt.xlabel("Classe prédite (valeur numérique)")
    plt.ylabel("Classe réelle (valeur numérique)")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_RESULTS, nom_png))
    plt.close()

# -------------------------------------------------------------
# 3. Random Forest
# -------------------------------------------------------------
print("\n--- 1. Random Forest ---")

params_rf = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20],
    "max_features": ["sqrt", 0.1]
}

print("Recherche des hyperparamètres sur 3000 images...")
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=seed),
    params_rf,
    cv=3,
    n_jobs=-1
)
grid_rf.fit(X_train_reduit, y_train_reduit)

print("Meilleurs paramètres RF :", grid_rf.best_params_)

print("Entraînement final sur tout le train (12000 images)...")
t0 = time.time()
clf_rf = grid_rf.best_estimator_
clf_rf.fit(X_train, y_train)
print(f"Fait en {round(time.time() - t0, 1)} secondes")

y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Score Random Forest sur le test : {acc_rf * 100:.2f} %")
print(classification_report(y_test, y_pred_rf, target_names=noms_classes, digits=3))

tracer_matrice(y_test, y_pred_rf, f"Random Forest ({acc_rf*100:.2f}%)", "confusion_random_forest.png")

# -------------------------------------------------------------
# 4. Réseau de Neurones (MLP)
# -------------------------------------------------------------
print("\n--- 2. Réseau de Neurones (MLP) ---")

params_mlp = {
    "hidden_layer_sizes": [(128,), (256,), (256, 128)],
    "alpha": [0.0001, 0.01],
    "learning_rate_init": [0.001, 0.003]
}

print("Recherche des hyperparamètres...")
grid_mlp = GridSearchCV(
    MLPClassifier(max_iter=200, early_stopping=True, random_state=seed),
    params_mlp,
    cv=3,
    n_jobs=-1
)
grid_mlp.fit(X_train_reduit, y_train_reduit)

print("Meilleurs paramètres MLP :", grid_mlp.best_params_)

print("Entraînement final sur tout le train (12000 images)...")
t0 = time.time()
clf_mlp = grid_mlp.best_estimator_
clf_mlp.fit(X_train, y_train)
print(f"Fait en {round(time.time() - t0, 1)} secondes")

y_pred_mlp = clf_mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Score MLP sur le test : {acc_mlp * 100:.2f} %")
print(classification_report(y_test, y_pred_mlp, target_names=noms_classes, digits=3))

tracer_matrice(y_test, y_pred_mlp, f"MLP ({acc_mlp*100:.2f}%)", "confusion_mlp.png")

# -------------------------------------------------------------
# 5. SVM
# -------------------------------------------------------------
print("\n--- 3. SVM ---")
params_svm = [
    {"kernel": ["rbf"], "C": [1, 10, 100], "gamma": ["scale", 0.001]},
    {"kernel": ["linear"], "C": [0.01, 1]}
]

print("Recherche des hyperparamètres SVM sur 3000 images...")
grid_svm = GridSearchCV(
    SVC(random_state=seed),
    params_svm,
    cv=3,
    n_jobs=-1
)
grid_svm.fit(X_train_reduit, y_train_reduit)

print("Meilleurs paramètres SVM :", grid_svm.best_params_)
print(f"Meilleur score CV (3000 images) : {grid_svm.best_score_ * 100:.2f} %")

print("Entraînement final sur tout le train (12000 images).")
t0 = time.time()
clf_svm = grid_svm.best_estimator_
clf_svm.fit(X_train, y_train)
print(f"Fait en {round(time.time() - t0, 1)} secondes")

acc_train_svm = clf_svm.score(X_train, y_train)
print(f"Score SVM sur le Train (12000 images) : {acc_train_svm * 100:.2f} %")

y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Score SVM sur le test : {acc_svm * 100:.2f} %")
print(classification_report(y_test, y_pred_svm, target_names=noms_classes, digits=3))

tracer_matrice(y_test, y_pred_svm, f"SVM ({acc_svm*100:.2f}%)", "confusion_svm.png")

# -------------------------------------------------------------
# 6. Réseau de Neurones Convolutif (CNN)
# -------------------------------------------------------------
print("\n--- 4. Réseau de Neurones Convolutif (CNN) ---")
print("On remet les images en 32x32x1 pour le CNN")

def creer_modele_cnn(lr, dropout_rate):
    # Construction du modèle couche par couche (plus classique)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    # On fixe les neurones à 128 pour cette recherche
    model.add(Dense(128, activation='relu'))
    
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(15, activation='softmax'))
    
    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Préparation des données pour le grid search (3000 images max pour gagner du temps)
X_red_cnn = X_train_reduit.reshape(-1, 32, 32, 1)
y_red_cnn = y_train_reduit - 1  # Keras veut des classes qui commencent à 0

X_sub_tr, X_sub_val, y_sub_tr, y_sub_val = train_test_split(
    X_red_cnn, y_red_cnn, test_size=0.20, random_state=seed, stratify=y_red_cnn
)

# Paramètres à tester
valeurs_lr = [0.01, 0.02]
valeurs_batch = [32, 64]
valeurs_dropout = [0.0, 0.1, 0.2]

print("Début du Grid Search manuel sur le CNN (12 combinaisons)...")
meilleur_score = 0.0
best_lr = 0.01
best_batch = 32
best_drop = 0.0

t0_grid = time.time()

for lr in valeurs_lr:
    for batch in valeurs_batch:
        for drop in valeurs_dropout:
            # Fixer le seed à chaque itération pour comparer proprement
            keras.utils.set_random_seed(seed)
            
            modele_test = creer_modele_cnn(lr=lr, dropout_rate=drop)
            
            # Entrainement court (12 epochs) pour tester vite fait
            modele_test.fit(X_sub_tr, y_sub_tr, epochs=12, batch_size=batch, verbose=0)
            
            # Evaluation sur le petit set de validation
            _, score_val = modele_test.evaluate(X_sub_val, y_sub_val, verbose=0)
            print(f"Test -> lr={lr}, batch={batch}, dropout={drop} | Score de val : {score_val*100:.2f}%")
            
            # Sauvegarde de la meilleure combinaison
            if score_val > meilleur_score:
                meilleur_score = score_val
                best_lr = lr
                best_batch = batch
                best_drop = drop

print(f"Recherche finie en {round(time.time() - t0_grid, 1)} sec")
print(f"Meilleurs paramètres trouvés : lr={best_lr}, batch_size={best_batch}, dropout={best_drop}")
print(f"Meilleur score de validation : {meilleur_score * 100:.2f} %")


# -------------------------------------------------------------
# Entraînement du vrai modèle final
# -------------------------------------------------------------
print("\nEntraînement du CNN final sur toutes les données (12000 images)...")
X_train_cnn = X_train.reshape(-1, 32, 32, 1)
X_test_cnn = X_test.reshape(-1, 32, 32, 1)

y_train_cnn = y_train - 1
y_test_cnn = y_test - 1

keras.utils.set_random_seed(seed)
modele_final = creer_modele_cnn(lr=best_lr, dropout_rate=best_drop)

t0 = time.time()
modele_final.fit(
    X_train_cnn, y_train_cnn,
    epochs=25,
    batch_size=best_batch,
    verbose=1
)
print(f"Entrainement terminé en {round(time.time() - t0, 1)} secondes")

# Prédictions sur le test (on remet +1 pour comparer avec les vrais labels de 1 à 15)
y_proba_cnn = modele_final.predict(X_test_cnn, verbose=0)
y_pred_cnn = np.argmax(y_proba_cnn, axis=1) + 1

acc_cnn = accuracy_score(y_test, y_pred_cnn)
print(f"Score final du CNN sur le jeu de test : {acc_cnn * 100:.2f} %")
print(classification_report(y_test, y_pred_cnn, target_names=noms_classes, digits=3))

tracer_matrice(y_test, y_pred_cnn, f"CNN ({acc_cnn*100:.2f}%)", "confusion_cnn.png")

# -------------------------------------------------------------
# 7. Bilan
# -------------------------------------------------------------
print("\n================ RÉSULTATS FINAUX ================")
print(f"Random Forest : {acc_rf * 100:.2f} %")
print(f"MLP           : {acc_mlp * 100:.2f} %")
print(f"SVM           : {acc_svm * 100:.2f} %")
print(f"CNN (2D)      : {acc_cnn * 100:.2f} %")
print("==================================================")
print("Les 4 matrices sont enregistrées dans le dossier results/.")