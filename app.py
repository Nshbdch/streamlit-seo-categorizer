# Fichier: app.py
import streamlit as st
import pandas as pd
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# --- Configuration de la page Streamlit ---
st.set_page_config(layout="wide", page_title="Clustering Mots-clés SEO (Simplifié)")

# --- Fonction pour extraire les top termes d'un cluster ---
def get_top_terms_per_cluster(tfidf_matrix, labels, feature_names, n_clusters, n_terms=5):
    cluster_terms = {}
    for i in range(n_clusters):
        # Indices des mots-clés dans ce cluster
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            cluster_terms[i] = "Cluster vide"
            continue
        # Vecteurs TF-IDF pour ce cluster
        cluster_vectors = tfidf_matrix[cluster_indices]
        # Moyenne des scores TF-IDF pour chaque terme dans ce cluster
        mean_tfidf_scores = np.array(cluster_vectors.mean(axis=0)).flatten()
        # Indices des N termes ayant les scores moyens les plus élevés
        top_term_indices = mean_tfidf_scores.argsort()[-n_terms:][::-1]
        # Noms des top termes
        top_terms = [feature_names[idx] for idx in top_term_indices]
        cluster_terms[i] = ", ".join(top_terms)
    return cluster_terms

# --- Interface Streamlit ---
st.title(" Outil de Clustering Sémantique (Simplifié) de Mots-clés SEO")
st.markdown("""
Collez votre liste de mots-clés ci-dessous (un mot-clé par ligne). L'outil tentera de les regrouper
en fonction des mots qu'ils partagent. Choisissez le nombre de groupes souhaité.
*Pour le mode sombre, cliquez sur ⚙️ > Settings > Theme: Dark*
""")

# Zone de texte pour l'input
keywords_input = st.text_area("Liste de mots-clés :", height=250, placeholder="lunette de soleil homme\nvoiture loa occasion bordeaux\nremboursement lunettes mutuelle\nlunettes de soleil femme tendance\nvoiture occasion pas cher\n...")

# Choix du nombre de clusters par l'utilisateur
n_clusters_requested = st.number_input(
    "Nombre de groupes (clusters) souhaité :",
    min_value=2,
    max_value=50, # Limite raisonnable
    value=5,      # Valeur par défaut
    step=1,
    help="Combien de groupes pensez-vous qu'il y a dans vos données ? L'algorithme essaiera de créer ce nombre de groupes."
)

# Bouton pour lancer le clustering
if st.button(" Regrouper les Mots-clés"):
    if keywords_input and n_clusters_requested:
        # Nettoyer et séparer les mots-clés
        keywords_list = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]

        if not keywords_list:
            st.warning("Veuillez entrer au moins un mot-clé.")
        elif len(keywords_list) < n_clusters_requested:
            st.warning(f"Vous avez demandé {n_clusters_requested} groupes, mais n'avez fourni que {len(keywords_list)} mots-clés. Réduisez le nombre de groupes.")
        else:
            try:
                # 1. Vectorisation TF-IDF
                # Utiliser des stop words français pour ignorer les mots courants
                # Vous pourriez avoir besoin d'installer un pack nltk: python -m nltk.downloader stopwords
                try:
                    from nltk.corpus import stopwords
                    french_stop_words = stopwords.words('french')
                except ImportError:
                    st.info("NLTK non trouvé pour les stop words français, utilisation d'une liste basique.")
                    french_stop_words = ['a', 'ai', 'aie', 'aient', 'aies', 'ait', 'alors', 'as', 'au', 'aucuns', 'aura', 'aurai', 'auraient', 'aurais', 'aurait', 'auras', 'aurez', 'auriez', 'aurions', 'aurons', 'auront', 'aussi', 'autre', 'aux', 'avaient', 'avais', 'avait', 'avant', 'avec', 'avez', 'aviez', 'avions', 'avons', 'ayant', 'ayante', 'ayantes', 'ayants', 'ayez', 'ayons', 'bon', 'car', 'ce', 'ceci', 'cela', 'ces', 'cet', 'cette', 'ceux', 'ceux-ci', 'ceux-là', 'chacun', 'chaque', 'ci', 'comme', 'comment', 'd', 'dans', 'de', 'des', 'deux', 'devoir', 'devrait', 'devront', 'dois', 'doit', 'donc', 'dos', 'droite', 'du', 'dès', 'début', 'elle', 'elles', 'en', 'encore', 'es', 'est', 'et', 'eu', 'eue', 'eues', 'eurent', 'eus', 'eusse', 'eussent', 'eusses', 'eussiez', 'eussions', 'eut', 'eux', 'eûmes', 'eût', 'eûtes', 'faire', 'fais', 'faisaient', 'faisais', 'faisait', 'faisant', 'fait', 'faites', 'faits', 'faura', 'faurai', 'fauras', 'faurez', 'fauront', 'femme', 'ferez', 'feriez', 'ferions', 'ferons', 'feront', 'fimes', 'firent', 'fis', 'fisse', 'fissent', 'fisses', 'fussiez', 'fussions', 'fit', 'furent', 'fus', 'fusse', 'fussent', 'fusses', 'fussiez', 'fussions', 'fut', 'fûmes', 'fût', 'fûtes', 'haut', 'homme', 'hors', 'ici', 'il', 'ils', 'j', 'je', 'juste', 'l', 'la', 'le', 'les', 'leur', 'leurs', 'lui', 'là', 'm', 'ma', 'maintenant', 'mais', 'mes', 'moi', 'moins', 'mon', 'mot', 'même', 'n', 'ne', 'ni', 'nom', 'nos', 'notre', 'nous', 'on', 'ont', 'ou', 'où', 'par', 'parce', 'pas', 'peut', 'peu', 'plupart', 'pour', 'pourquoi', 'qu', 'quand', 'que', 'quel', 'quelle', 'quelles', 'quels', 'qui', 's', 'sa', 'sans', 'se', 'sera', 'serai', 'seraient', 'serais', 'serait', 'seras', 'serez', 'seriez', 'serions', 'serons', 'seront', 'ses', 'seulement', 'si', 'soi', 'soient', 'sois', 'soit', 'sommes', 'son', 'sont', 'sous', 'soyez', 'soyons', 'suis', 'sur', 't', 'ta', 'tandis', 'te', 'tellement', 'tels', 'tes', 'toi', 'ton', 'tous', 'tout', 'trop', 'très', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'vu', 'y', 'à', 'ça', 'étaient', 'étais', 'était', 'étant', 'étante', 'étantes', 'étants', 'étiez', 'étions', 'étés', 'étée', 'étées', 'étés', 'êtes', 'être'] # Liste basique si nltk n'est pas là


                vectorizer = TfidfVectorizer(stop_words=french_stop_words,
                                             lowercase=True,
                                             max_df=0.9,  # Ignore terms that appear in > 90% of keywords
                                             min_df=2 if len(keywords_list) > 10 else 1, # Ignore terms that appear less than 2 times (if enough data)
                                             ngram_range=(1, 2)) # Considère les mots seuls et les paires de mots
                tfidf_matrix = vectorizer.fit_transform(keywords_list)
                feature_names = vectorizer.get_feature_names_out()

                # 2. Clustering K-Means
                kmeans = KMeans(n_clusters=n_clusters_requested,
                                random_state=42, # Pour la reproductibilité
                                n_init='auto') # Version moderne pour éviter les warnings
                kmeans.fit(tfidf_matrix)
                labels = kmeans.labels_

                # 3. Création du DataFrame de résultats
                df_results = pd.DataFrame({'Mot-clé': keywords_list, 'ID Groupe': labels})
                df_results = df_results.sort_values(by='ID Groupe') # Trier par groupe pour la lisibilité

                # 4. Obtenir les termes représentatifs de chaque groupe
                cluster_descriptions = get_top_terms_per_cluster(tfidf_matrix, labels, feature_names, n_clusters_requested)

                # Afficher les résultats
                st.subheader("Résultats du Clustering :")
                st.dataframe(df_results, use_container_width=True)

                st.subheader("Thèmes des Groupes (Mots Représentatifs) :")
                for group_id, terms in cluster_descriptions.items():
                    st.markdown(f"**Groupe {group_id}:** `{terms}`")

                # Préparer le fichier Excel pour le téléchargement
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Clustering SEO')
                    # Ajouter une feuille avec les descriptions des groupes
                    df_descriptions = pd.DataFrame(list(cluster_descriptions.items()), columns=['ID Groupe', 'Termes Représentatifs'])
                    df_descriptions.to_excel(writer, index=False, sheet_name='Descriptions Groupes')

                excel_data = output.getvalue()

                # Bouton de téléchargement Excel
                st.download_button(
                    label=" Télécharger les résultats (Excel)",
                    data=excel_data,
                    file_name='clustering_mots_cles_seo.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            except Exception as e:
                st.error(f"Une erreur est survenue lors du traitement : {e}")
                st.error("Cela peut arriver si les mots-clés sont très courts, très similaires, ou si le nombre de groupes demandé est inadapté.")

    elif not keywords_input:
        st.warning("Veuillez coller des mots-clés dans la zone de texte.")

st.sidebar.info("""
**Comment ça marche (Simplifié) ?**
1.  Les mots-clés sont transformés en vecteurs basés sur l'importance des mots qu'ils contiennent (TF-IDF), en ignorant les mots très courants (stop words).
2.  L'algorithme K-Means regroupe les mots-clés ayant les vecteurs les plus similaires.
3.  Le nombre de groupes est défini par vous.
4.  Chaque groupe est décrit par les mots qui y sont les plus caractéristiques (score TF-IDF moyen le plus élevé).

**Limites :**
- Le choix du "Nombre de groupes" est crucial et influe sur le résultat.
- Ne comprend pas la sémantique profonde (ex: synonymes). Le regroupement est basé sur les mots *exacts* partagés (après nettoyage).
- Les "Termes Représentatifs" sont indicatifs, pas un nom de catégorie parfait.
""")
