import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 1. Charger les données
df = pd.read_csv("chinese_mnist.csv")

# 2. Récupérer la première image de chaque classe (code de 1 à 15)
classes_uniques = df.drop_duplicates(subset=['code']).sort_values('code')

# 3. Créer une grille 3 lignes x 5 colonnes
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
axes = axes.flatten()

for idx, (index, row) in enumerate(classes_uniques.iterrows()):
    suite = int(row["suite_id"])
    sample = int(row["sample_id"])
    code = int(row["code"])
    valeur = str(row["value"])
    
    # Reconstruire le chemin de l'image
    nom_fichier = f"input_{suite}_{sample}_{code}.jpg"
    chemin_image = os.path.join("data", nom_fichier)
    
    # Ouvrir et afficher
    img = Image.open(chemin_image).convert("L")
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f"Valeur : {valeur}", fontsize=10)
    axes[idx].axis('off')

plt.tight_layout()
# Sauvegarder dans le dossier results
plt.savefig(os.path.join("results", "apercu_dataset.png"), dpi=150)
print("Image 'apercu_dataset.png' générée dans le dossier results/ !")