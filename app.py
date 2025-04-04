# Fichier: app.py
import streamlit as st
import pandas as pd
import io
import re # Pour des recherches un peu plus flexibles

# --- Configuration de la page Streamlit ---
st.set_page_config(layout="wide", page_title="Catégorisation Mots-clés SEO")

# --- Fonction de Catégorisation (Basée sur des règles) ---
# NOTE : Cette fonction est une ébauche. Il faudra l'enrichir considérablement
# avec VOS règles métier spécifiques, termes, marques, localisations etc.
def categorize_keyword(keyword):
    # Initialisation des catégories avec des valeurs par défaut
    categories = {
        'Mot-clé': keyword,
        'Typologie de requête': '-',
        'Catégorie': '-',
        'Sous-catégorie': '-',
        'Cible': 'Générique',
        'Financement': '-',
        'Localisation': '-',
        'Type/Etat': '-'
    }
    keyword_lower = keyword.lower() # Travailler en minuscules pour la comparaison

    # 1. Typologie de requête (Exemples)
    if any(term in keyword_lower for term in ["prix", "acheter", "achat", "commander", "pas cher", "boutique", "shop", "promo", "soldes"]):
        categories['Typologie de requête'] = 'E-commerce'
    elif any(term in keyword_lower for term in ["comment", "definition", "avis", "test", "comparatif", "guide", "information", "blog", "symptome", "probleme"]):
        categories['Typologie de requête'] = 'Blog/Info'
    elif "remboursement" in keyword_lower or "mutuelle" in keyword_lower:
         categories['Typologie de requête'] = 'Blog/Info' # Ou autre selon votre logique

    # 2. Catégorie Principale (Exemples)
    if "lunette" in keyword_lower: categories['Catégorie'] = 'Lunette'
    if "solaire" in keyword_lower: categories['Catégorie'] = 'Solaire' # Peut écraser "Lunette" si les deux sont présents
    if "verre progressif" in keyword_lower or "verre polarisant" in keyword_lower: categories['Catégorie'] = 'Verre'
    if any(term in keyword_lower for term in ["voiture", "auto", "vehicule", "automobile"]): categories['Catégorie'] = 'Automobile'
    if any(term in keyword_lower for term in ["santé", "yeux", "hypermetrope", "ametropie", "opticien"]): categories['Catégorie'] = 'Santé/Optique'

    # 3. Sous-catégorie (Exemples)
    # Marques (Exemple Optique)
    marques_optique = ["cartier", "chloé", "fendi", "oakley", "prada", "ray ban", "chanel", "dior", "louis vuitton"]
    for marque in marques_optique:
        if marque in keyword_lower:
            categories['Sous-catégorie'] = 'Marque produit'
            break # Stop dès qu'une marque est trouvée

    # Formes/Types (Exemple Optique)
    if any(term in keyword_lower for term in ["ski", "sport"]): categories['Sous-catégorie'] = 'Sport/Ski'
    elif any(term in keyword_lower for term in ["aviateur", "carré", "rectangulaire", "ronde"]): categories['Sous-catégorie'] = 'Forme'
    elif any(term in keyword_lower for term in ["photochromique", "polarisante"]): categories['Sous-catégorie'] = 'Technologie'
    elif "remboursement" in keyword_lower or "mutuelle" in keyword_lower: categories['Sous-catégorie'] = 'Mutuelle'
    elif "prix" in keyword_lower or "pas cher" in keyword_lower: categories['Sous-catégorie'] = 'Prix'

    # 4. Cible (Exemples)
    if "homme" in keyword_lower: categories['Cible'] = 'Homme'
    if "femme" in keyword_lower: categories['Cible'] = 'Femme'
    if "enfant" in keyword_lower: categories['Cible'] = 'Enfant' # Ajouter si pertinent

    # 5. Financement (Exemple Automobile)
    if "loa" in keyword_lower or "leasing" in keyword_lower: categories['Financement'] = 'LOA/Leasing'
    if "lld" in keyword_lower: categories['Financement'] = 'LLD'
    if "credit" in keyword_lower: categories['Financement'] = 'Crédit'

    # 6. Localisation (Exemples) - Utilise regex pour trouver des noms de villes communs
    # C'est très basique, une vraie solution nécessiterait une liste de villes/régions
    locations = re.findall(r'\b(paris|lyon|marseille|bordeaux|toulouse|nice|nantes|strasbourg|montpellier|lille)\b', keyword_lower)
    if locations:
        categories['Localisation'] = locations[0].capitalize() # Prend la première trouvée

    # 7. Type/Etat (Exemple Automobile)
    if "occasion" in keyword_lower: categories['Type/Etat'] = 'Occasion'
    if "neuf" in keyword_lower or "neuve" in keyword_lower: categories['Type/Etat'] = 'Neuf'

    # --- Ajoutez ici d'autres règles pour affiner la catégorisation ---
    # Par exemple :
    # if categories['Catégorie'] == 'Solaire' and categories['Sous-catégorie'] == '-':
    #     categories['Sous-catégorie'] = 'Générique Solaire' # Règle par défaut si rien d'autre trouvé

    return categories

# --- Interface Streamlit ---
st.title(" Outil de Catégorisation Sémantique de Mots-clés SEO")
st.markdown("""
Collez votre liste de mots-clés ci-dessous (un mot-clé par ligne). L'outil tentera de les catégoriser
selon des règles prédéfinies. Cliquez ensuite sur le bouton pour générer le tableau et l'exporter.
*Pour le mode sombre, cliquez sur ⚙️ > Settings > Theme: Dark*
""")

# Zone de texte pour l'input
keywords_input = st.text_area("Liste de mots-clés :", height=250, placeholder="lunette de soleil homme\nvoiture loa occasion bordeaux\nremboursement lunettes mutuelle\n...")

# Bouton pour lancer la catégorisation
if st.button(" Catégoriser les Mots-clés"):
    if keywords_input:
        # Nettoyer et séparer les mots-clés
        keywords_list = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]

        if not keywords_list:
            st.warning("Veuillez entrer au moins un mot-clé.")
        else:
            # Appliquer la fonction de catégorisation à chaque mot-clé
            results_list = [categorize_keyword(kw) for kw in keywords_list]

            # Créer un DataFrame Pandas
            df_results = pd.DataFrame(results_list)

            # Afficher le DataFrame dans Streamlit
            st.subheader("Résultats de la Catégorisation :")
            st.dataframe(df_results, use_container_width=True) # use_container_width pour prendre toute la largeur

            # Préparer le fichier Excel pour le téléchargement
            output = io.BytesIO()
            # Utilisation de 'with' garantit que le writer est correctement finalisé
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Categorisation SEO')
            excel_data = output.getvalue()

            # Bouton de téléchargement Excel
            st.download_button(
                label=" Télécharger les résultats (Excel)",
                data=excel_data,
                file_name='categorisation_mots_cles_seo.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    else:
        st.warning("Veuillez coller des mots-clés dans la zone de texte.")

st.sidebar.info("""
**Comment ça marche ?**
Cet outil utilise des règles prédéfinies (mots spécifiques, marques, localisations...) pour assigner des catégories à vos mots-clés.

**Limites :**
- La précision dépend de l'exhaustivité des règles.
- Pas de compréhension profonde du langage.
- Les règles peuvent nécessiter des ajustements pour VOS besoins spécifiques.

**Personnalisation :**
Le code source (fichier `.py`) peut être modifié pour ajouter/modifier les règles de catégorisation.
""")