import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import os
import json
from datetime import datetime
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import openai
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyseur de reddition de charges", layout="wide")

# Configuration des secrets (Streamlit Cloud)
@st.cache_resource
def initialize_openai():
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return client

# =============== FONCTIONS UTILITAIRES ===============

def clean_monetary_value(value):
    """Convertit une chaîne représentant un montant en valeur numérique"""
    if not value or not isinstance(value, (str, int, float)):
        return 0.0
        
    if isinstance(value, (int, float)):
        return float(value)
        
    # Supprimer les symboles de devise et les espaces
    value = str(value).replace('€', '').replace(' ', '').strip()
    
    # Remplacer la virgule par un point pour les décimales
    value = value.replace(',', '.')
    
    # Gérer les cas négatifs avec parenthèses (2 499,55) -> -2499.55
    if '(' in value and ')' in value:
        value = value.replace('(', '').replace(')', '')
        value = '-' + value
    
    try:
        return float(value)
    except ValueError:
        return 0.0

def prepare_charges_for_chart(data):
    """Prépare les données des charges pour la visualisation"""
    detail_charges = data.get('Détail des charges par catégorie', [])
    
    if not detail_charges:
        return pd.DataFrame()
    
    # Créer un DataFrame avec les catégories et montants
    chart_data = []
    for charge in detail_charges:
        montant = clean_monetary_value(charge.get('montant TTC', 0))
        if montant > 0:  # Ignorer les montants nuls ou négatifs pour le graphique
            chart_data.append({
                'Catégorie': charge.get('poste de charge', 'Autre'),
                'Montant': montant
            })
    
    df = pd.DataFrame(chart_data)
    
    # Regrouper les petites catégories sous "Autres" pour un graphique plus lisible
    if len(df) > 7:
        # Trier par montant décroissant
        df = df.sort_values('Montant', ascending=False)
        
        # Conserver les 6 premières catégories et regrouper les autres
        top_categories = df.head(6)
        other_categories = df.tail(len(df) - 6)
        
        other_sum = other_categories['Montant'].sum()
        
        # Créer un nouveau DataFrame avec les catégories principales et "Autres"
        if other_sum > 0:
            new_df = pd.concat([
                top_categories,
                pd.DataFrame([{'Catégorie': 'Autres', 'Montant': other_sum}])
            ])
            df = new_df
    
    return df

# =============== FONCTIONS D'EXTRACTION ET D'ANALYSE ===============

# Fonction pour extraire le texte des images
def extract_text_from_images(images):
    extracted_text = ""
    for img in images:
        text = pytesseract.image_to_string(img, lang='fra')
        extracted_text += text + "\n\n"
    return extracted_text

# Fonction pour convertir PDF en images
def convert_pdf_to_images(pdf_file):
    try:
        images = convert_from_bytes(pdf_file.getvalue())
        return images
    except Exception as e:
        st.error(f"Erreur lors de la conversion du PDF: {e}")
        return None

# Prompt pour l'extraction d'informations
DETAILED_PROMPT = """
Tu es un expert en analyse de documents financiers français, spécialisé dans les redditions de charges locatives. Analyse ce document et extrait les informations structurées précises.

Document à analyser:
{text}

Retourne un JSON structuré avec ces informations précises:
1. Information du propriétaire/gestionnaire (nom, adresse complète)
2. Information du locataire/occupant (nom, adresse, référence)
3. Période concernée (dates début et fin)
4. Montant total des provisions versées
5. Montant total des charges réelles
6. Solde (montant, type: créditeur ou débiteur)
7. Détail des charges par catégorie (avec pour chaque poste: poste de charge, montant HT, TVA, montant TTC)
8. Références du document (numéro, date d'émission)

Pour les montants:
- Extrait les valeurs numériques précises
- Précise si les montants incluent la TVA
- Le solde doit indiquer s'il est en faveur du locataire (créditeur) ou à payer (débiteur)

Les montants doivent être extraits comme des nombres à virgule (ex: 2499.55).
"""

# Fonction pour analyser le texte avec GPT
def analyze_with_gpt(client, text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en analyse de documents de reddition de charges locatives en France. Extrais les informations précises demandées."},
                {"role": "user", "content": DETAILED_PROMPT.format(text=text)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse avec GPT: {e}")
        return None

# =============== INTERFACE UTILISATEUR ===============

st.title("Analyseur de reddition de charges locatives")

with st.expander("ℹ️ À propos de cet outil", expanded=True):
    st.markdown("""
    Cet outil analyse les documents de reddition de charges locatives en utilisant OCR et IA.
    1. Téléchargez un document PDF de reddition de charges
    2. L'OCR extrait le texte des images
    3. GPT-4o-mini analyse le texte et extrait les informations structurées
    4. Les résultats sont présentés sous forme de tableaux et graphiques interactifs
    """)

uploaded_file = st.file_uploader("Téléchargez un document de reddition de charges (PDF)", type=["pdf"])

if uploaded_file is not None:
    try:
        # Initialisation du client OpenAI avec les secrets Streamlit
        client = initialize_openai()
        
        with st.spinner("Traitement du document en cours..."):
            # Conversion du PDF en images
            images = convert_pdf_to_images(uploaded_file)
            
            if images:
                # Affichage des pages du document
                tabs = st.tabs([f"Page {i+1}" for i in range(len(images))])
                for i, tab in enumerate(tabs):
                    with tab:
                        st.image(images[i], caption=f"Page {i+1}", use_column_width=True)
                
                # Extraction du texte
                with st.spinner("Extraction du texte..."):
                    extracted_text = extract_text_from_images(images)
                    
                    with st.expander("Texte brut extrait (pour débogage)", expanded=False):
                        st.text_area("", extracted_text, height=200)
                
                # Analyse avec GPT
                with st.spinner("Analyse des informations avec IA..."):
                    analysis_result = analyze_with_gpt(client, extracted_text)
                    
                    if analysis_result:
                        # Affichage des résultats structurés
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Informations générales")
                            st.write(f"**Propriétaire/Gestionnaire:** {analysis_result.get('Information du propriétaire/gestionnaire', {}).get('nom', 'Non détecté')}")
                            st.write(f"**Adresse:** {analysis_result.get('Information du propriétaire/gestionnaire', {}).get('adresse', 'Non détectée')}")
                            st.write(f"**Période:** {analysis_result.get('Période concernée', {}).get('début', 'Non détectée')} au {analysis_result.get('Période concernée', {}).get('fin', 'Non détectée')}")
                            st.write(f"**Référence document:** {analysis_result.get('Références du document', {}).get('numéro', 'Non détectée')}")
                            st.write(f"**Date d'émission:** {analysis_result.get('Références du document', {}).get('date', 'Non détectée')}")
                        
                        with col2:
                            st.subheader("Montants")
                            
                            # Extraire et formater les montants
                            provisions = clean_monetary_value(analysis_result.get('Montant total des provisions versées', 0))
                            charges = clean_monetary_value(analysis_result.get('Montant total des charges réelles', 0))
                            
                            solde_info = analysis_result.get('Solde', {})
                            if isinstance(solde_info, dict):
                                solde_value = clean_monetary_value(solde_info.get('montant', 0))
                                solde_type = solde_info.get('type', '')
                            else:
                                solde_value = clean_monetary_value(solde_info)
                                solde_type = 'Non précisé'
                            
                            # Afficher les métriques
                            st.metric("Provisions versées", f"{provisions:,.2f} €".replace(',', ' '))
                            st.metric("Charges réelles", f"{charges:,.2f} €".replace(',', ' '))
                            
                            delta_color = "normal"
                            if "créditeur" in str(solde_type).lower() or "en votre faveur" in str(solde_type).lower():
                                delta_color = "inverse"
                            
                            st.metric("Solde", f"{solde_value:,.2f} €".replace(',', ' '), 
                                     delta=solde_type, 
                                     delta_color=delta_color)
                        
                        # Détail des charges sous forme de tableau
                        st.subheader("Détail des charges")
                        
                        charges_detail = analysis_result.get('Détail des charges par catégorie', [])
                        if charges_detail and isinstance(charges_detail, list):
                            # Création d'un DataFrame pour afficher les charges
                            charges_df = pd.DataFrame(charges_detail)
                            st.dataframe(charges_df, use_container_width=True)
                            
                            # Visualisation des charges
                            try:
                                chart_data = prepare_charges_for_chart(analysis_result)
                                if not chart_data.empty:
                                    st.subheader("Répartition des charges")
                                    fig = px.bar(chart_data, x='Catégorie', y='Montant', 
                                               title='Répartition des charges par catégorie',
                                               color='Catégorie')
                                    fig.update_layout(xaxis_title='Catégorie', 
                                                    yaxis_title='Montant (€)',
                                                    xaxis_tickangle=-45)
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Impossible de générer le graphique : {e}")
                        else:
                            st.warning("Aucun détail des charges n'a été extrait ou le format n'est pas reconnu.")
                        
                        # Affichage du JSON complet
                        with st.expander("Résultat complet (JSON)", expanded=False):
                            st.json(analysis_result)
                        
                        # Export des données
                        st.download_button(
                            label="Télécharger l'analyse (JSON)",
                            data=json.dumps(analysis_result, indent=2, ensure_ascii=False),
                            file_name=f"analyse_reddition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:
                        st.error("Échec de l'analyse du document. Veuillez réessayer avec un autre document.")
            else:
                st.error("Impossible de traiter le PDF. Vérifiez que le fichier est valide.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        st.info("Vérifiez que la clé API OpenAI est correctement configurée dans les secrets Streamlit.")

# Ajout d'un footer
st.markdown("---")
st.write("Développé avec Streamlit, Tesseract OCR et GPT-4o-mini")
