import streamlit as st
import os
import pandas as pd
import numpy as np
import pytesseract
import cv2
from PIL import Image
import json
import csv
from openai import OpenAI
import tempfile
import io
from pathlib import Path
import base64
import re

# Titre et configuration de la page
st.set_page_config(
    page_title="Analyseur de Charges - Baux Commerciaux",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .results-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ChargesAnalyzer:
    """
    Classe pour analyser les tableaux de charges des baux commerciaux 
    à partir de différents formats (Excel, CSV, PDF) en utilisant l'OCR et GPT-4o-mini.
    """
    
    def __init__(self, api_key=None, ocr_api_key=None, use_ai=True, tesseract_config=None):
        """
        Initialise l'analyseur de charges.
        
        Args:
            api_key (str, optional): Clé API OpenAI.
            ocr_api_key (str, optional): Clé API OCR (si utilisation d'une API OCR externe).
            use_ai (bool, optional): Indique si l'IA doit être utilisée pour l'analyse.
            tesseract_config (str, optional): Configuration Tesseract OCR personnalisée.
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Clé API OpenAI non fournie.")
            
        # Initialiser le client OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        # Stocker la clé API OCR si fournie
        self.ocr_api_key = ocr_api_key
        
        # Utilisation de l'IA pour l'analyse
        self.use_ai = use_ai
        
        # Configuration de Tesseract OCR pour Streamlit Cloud
        # Dans Streamlit Cloud, nous devons utiliser pytesseract avec des configurations spécifiques
        self.tesseract_config = tesseract_config or r'--oem 3 --psm 6'
    
    def process_file(self, uploaded_file):
        """
        Traite un fichier téléchargé par l'utilisateur.
        
        Args:
            uploaded_file (UploadedFile): Fichier téléchargé via Streamlit.
            
        Returns:
            dict: Résultats de l'analyse structurés.
        """
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Écrire le contenu du fichier téléchargé dans le fichier temporaire
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == '.pdf':
                result = self.analyze_pdf(tmp_path)
            elif file_extension in ['.xlsx', '.xls']:
                result = self.analyze_excel(tmp_path)
            elif file_extension == '.csv':
                result = self.analyze_csv(tmp_path)
            else:
                raise ValueError(f"Format de fichier non supporté: {file_extension}")
            
            # Supprimer le fichier temporaire
            os.unlink(tmp_path)
            
            return result
        except Exception as e:
            # Supprimer le fichier temporaire en cas d'erreur
            os.unlink(tmp_path)
            raise e
    
    def analyze_pdf(self, pdf_path):
        """
        Analyse un fichier PDF en utilisant l'OCR et GPT-4o-mini.
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF.
            
        Returns:
            dict: Résultats de l'analyse.
        """
        st.info("Analyse du PDF en cours... Cette opération peut prendre quelques instants.")
        progress_bar = st.progress(0)
        
        try:
            # Utiliser pdf2image pour convertir le PDF en images
            from pdf2image import convert_from_path
            
            # Sur Streamlit Cloud, nous devons spécifier le chemin vers poppler
            try:
                images = convert_from_path(pdf_path)
            except Exception as e:
                st.error(f"Erreur lors de la conversion du PDF: {str(e)}")
                st.warning("Tentative d'installation de poppler...")
                
                # Afficher une information pour l'installation manuelle de poppler dans requirements.txt
                st.info("Assurez-vous que 'poppler-utils' est installé sur votre serveur Streamlit Cloud. Ajoutez les dépendances dans requirements.txt")
                return {"error": "L'installation de poppler est requise pour traiter les PDFs."}
        
            extracted_text = ""
            total_pages = len(images)
            
            for i, image in enumerate(images):
                # Mettre à jour la barre de progression
                progress_bar.progress((i + 0.5) / total_pages)
                
                # Prétraitement de l'image pour améliorer l'OCR
                img_np = np.array(image)
                
                # Conversion en niveaux de gris
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                
                # Appliquer un seuillage pour améliorer la netteté
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                # Effectuer l'OCR
                page_text = pytesseract.image_to_string(binary, config=self.tesseract_config, lang='fra')
                
                # Ajouter le texte extrait
                extracted_text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
                
                # Mettre à jour la barre de progression
                progress_bar.progress((i + 1) / total_pages)
        
            # Analyser le texte extrait avec GPT-4o-mini
            return self._analyze_with_gpt(extracted_text, source_type="PDF")
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse du PDF: {str(e)}")
            return {"error": str(e)}
    
    def analyze_excel(self, excel_path):
        """
        Analyse un fichier Excel.
        
        Args:
            excel_path (str): Chemin vers le fichier Excel.
            
        Returns:
            dict: Résultats de l'analyse.
        """
        st.info("Analyse du fichier Excel en cours...")
        
        try:
            # Lire toutes les feuilles du fichier Excel
            excel_data = {}
            xlsx = openpyxl.load_workbook(excel_path, data_only=True)
            
            for sheet_name in xlsx.sheetnames:
                sheet = xlsx[sheet_name]
                
                # Convertir la feuille en texte
                sheet_text = f"Feuille: {sheet_name}\n\n"
                
                for row in sheet.iter_rows(values_only=True):
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    sheet_text += " | ".join(row_values) + "\n"
                
                excel_data[sheet_name] = sheet_text
            
            # Concaténer toutes les feuilles en un seul texte
            combined_text = "\n\n".join([f"--- {sheet} ---\n{text}" for sheet, text in excel_data.items()])
            
            # Analyser avec GPT-4o-mini
            return self._analyze_with_gpt(combined_text, source_type="Excel")
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse du fichier Excel: {str(e)}")
            return {"error": str(e)}
    
    def analyze_csv(self, csv_path):
        """
        Analyse un fichier CSV.
        
        Args:
            csv_path (str): Chemin vers le fichier CSV.
            
        Returns:
            dict: Résultats de l'analyse.
        """
        st.info("Analyse du fichier CSV en cours...")
        
        try:
            # Essayer de déterminer le délimiteur
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as file:
                sample = file.read(1024)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            
            # Lire le CSV
            df = pd.read_csv(csv_path, delimiter=delimiter, encoding='utf-8', errors='replace')
            
            # Convertir en texte
            csv_text = df.to_string(index=False)
            
            # Analyser avec GPT-4o-mini
            return self._analyze_with_gpt(csv_text, source_type="CSV")
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse du fichier CSV: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_with_gpt(self, text, source_type):
        """
        Utilise GPT-4o-mini pour analyser le texte extrait.
        
        Args:
            text (str): Texte à analyser.
            source_type (str): Type de la source (PDF, Excel, CSV).
            
        Returns:
            dict: Résultats de l'analyse structurés.
        """
        logger.info(f"Analyse du texte avec GPT-4o-mini (source: {source_type})")
        
        # Limiter la taille du texte si nécessaire
        if len(text) > 15000:
            logger.warning(f"Le texte a été tronqué de {len(text)} à 15000 caractères")
            text = text[:15000] + "...[tronqué]"
        
        # Définir le prompt pour GPT-4o-mini
        prompt = f"""
        Tu es un expert en analyse de tableaux de charges et redditions de charges pour des baux commerciaux.
        Voici un tableau de charges extrait d'un document {source_type}.
        
        {text}
        
        Analyse précisément ce tableau de charges et extrait les informations suivantes au format JSON :
        1. Type de document : détermine s'il s'agit d'une reddition de charges, d'un appel de charges ou d'un budget prévisionnel
        2. Année ou période concernée : extrait la période exacte mentionnée (format JJ/MM/AAAA)
        3. Montant total des charges : montant total en euros
        4. Répartition des charges par poste : extrait tous les postes mentionnés (nettoyage, sécurité, maintenance, etc.) avec leurs montants
        5. Quote-part du locataire : fraction ou pourcentage des charges affectées au locataire (tantièmes)
        6. Montant facturé au locataire : montant final à payer par le locataire
        7. Montant des provisions déjà versées : sommes déjà payées par le locataire
        8. Solde : différence entre les provisions et les charges réelles (créditeur ou débiteur)
        
        Sois particulièrement attentif aux montants précédés de signes négatifs (-) qui indiquent des crédits ou des provisions.
        Pour les redditions de charges, le solde peut être positif (à la charge du locataire) ou négatif (en faveur du locataire).
        
        Réponds uniquement avec le JSON structuré des résultats.
        """
        
        try:
            # Tenter d'abord d'extraire les informations sans appeler l'API
            # Particulièrement utile pour les formats standards comme les redditions de charges
            result = self._extract_charges_data(text)
            
            # Si la méthode standard a donné des résultats satisfaisants, les utiliser
            if self._is_extraction_complete(result):
                st.success("Analyse réussie avec l'extraction directe!")
                return result
                
            # Sinon, utiliser l'API OpenAI
            st.info("Utilisation de l'IA pour compléter l'analyse...")
            
            # Appeler l'API OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé dans l'analyse de documents financiers immobiliers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Extraire la réponse
            analysis_text = response.choices[0].message.content
            
            # Nettoyer la réponse pour extraire uniquement le JSON
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = analysis_text[json_start:json_end]
                try:
                    # Analyser le JSON
                    analysis_result = json.loads(json_text)
                    st.success("Analyse réussie avec l'IA!")
                    return analysis_result
                except json.JSONDecodeError as e:
                    st.error(f"Erreur de décodage JSON: {str(e)}")
                    return {"error": "Format de réponse invalide", "raw_response": analysis_text}
            else:
                st.error("Pas de JSON trouvé dans la réponse")
                return {"error": "Pas de JSON trouvé dans la réponse", "raw_response": analysis_text}
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")
            return {"error": str(e)}
            
    def _extract_charges_data(self, text):
        """
        Méthode d'extraction directe des données de charges sans recourir à l'API.
        Particulièrement utile pour les formats standards comme les redditions de charges.
        
        Args:
            text (str): Le texte extrait du document.
            
        Returns:
            dict: Les informations extraites.
        """
        # Déterminer le type de document
        type_document = "Non spécifié"
        if "REDDITION" in text or "reddition" in text.lower():
            type_document = "Reddition de charges"
        elif "APPEL DE CHARGES" in text or "appel de charges" in text.lower():
            type_document = "Appel de charges"
        elif "BUDGET PRÉVISIONNEL" in text or "budget prévisionnel" in text.lower():
            type_document = "Budget prévisionnel"
            
        # Extraire la période concernée
        periode = "Non spécifiée"
        periode_pattern = r'[Pp]ériode du (\d{2}/\d{2}/\d{4}) au (\d{2}/\d{2}/\d{4})'
        periode_match = re.search(periode_pattern, text)
        if periode_match:
            periode = f"{periode_match.group(1)} au {periode_match.group(2)}"
            
        # Extraire le montant total des charges
        montant_total = "Non spécifié"
        # Plusieurs patterns possibles selon le format du document
        patterns_montant = [
            r'Total charges\s+([0-9\s]+[,.]\d{2})',
            r'Charges\s+([0-9\s]+[,.]\d{2}\s*€?)',
            r'REDDITION CHARGES[^\n]+\s+([0-9\s]+[,.]\d{2}\s*€?)'
        ]
        
        for pattern in patterns_montant:
            match = re.search(pattern, text)
            if match:
                montant_total = match.group(1).replace(' ', '').replace('€', '')
                break
                
        # Extraire le montant des provisions versées
        provisions = "Non spécifié"
        provisions_pattern = r'Provisions\s+(-?[0-9\s]+[,.]\d{2}\s*€?)'
        provisions_match = re.search(provisions_pattern, text)
        if provisions_match:
            provisions = provisions_match.group(1).replace(' ', '').replace('€', '')
            
        # Extraire le solde
        solde = "Non spécifié"
        solde_pattern = r'Solde\s+(-?[0-9\s]+[,.]\d{2}\s*€?)'
        solde_match = re.search(solde_pattern, text)
        if solde_match:
            solde = solde_match.group(1).replace(' ', '').replace('€', '')
            
        # Extraire la quote-part du locataire
        quote_part = "Non spécifiée"
        if "Tantièmes" in text and "Quote-part" in text:
            # Rechercher les tantièmes spécifiques au locataire
            tantiemes_pattern = r'(\d+)\s+(\d+\.\d{2})\s+365\s+'
            tantiemes_match = re.search(tantiemes_pattern, text)
            if tantiemes_match:
                quote_part = f"{tantiemes_match.group(1)}/{tantiemes_match.group(2)} (365 jours)"
                
        # Extraire la répartition des charges par poste
        repartition_charges = {}
        charges_pattern = r'(\d{2} / \d{2}|\d{2}/\d{2})\s+([A-Z\s/&.]+)(?:\s+)([0-9\s]+[,.]\d{2}\s*€?)(?:\s+\d+\s+[\d.]+\s+\d+\s+)([0-9\s]+[,.]\d{2})'
        
        for match in re.finditer(charges_pattern, text):
            poste = match.group(2).strip()
            montant = match.group(4).replace(' ', '')
            repartition_charges[poste] = montant
            
        # Extraire le montant facturé au locataire
        montant_facture = "Non spécifié"
        if type_document == "Reddition de charges" and solde != "Non spécifié":
            # Pour une reddition, le montant facturé est le solde
            montant_facture = solde
            
        result = {
            "Type de document": type_document,
            "Année ou période concernée": periode,
            "Montant total des charges": montant_total,
            "Répartition des charges par poste": repartition_charges,
            "Quote-part du locataire": quote_part,
            "Montant facturé au locataire": montant_facture,
            "Montant des provisions déjà versées": provisions,
            "Solde": solde
        }
        
        return result
        
    def _is_extraction_complete(self, result):
        """
        Vérifie si l'extraction directe a donné des résultats satisfaisants.
        
        Args:
            result (dict): Les résultats de l'extraction.
            
        Returns:
            bool: True si l'extraction est complète, False sinon.
        """
        # Vérifier que les champs essentiels contiennent des informations
        essential_fields = [
            "Type de document", 
            "Montant total des charges", 
            "Solde"
        ]
        
        for field in essential_fields:
            if field not in result or result[field] == "Non spécifié":
                return False
                
        # Vérifier qu'il y a au moins quelques postes de charges
        if "Répartition des charges par poste" not in result or len(result["Répartition des charges par poste"]) < 2:
            return False
            
        return True

def download_json(data):
    """Génère un lien de téléchargement pour les résultats au format JSON."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="resultats_analyse.json">Télécharger les résultats (JSON)</a>'
    return href

def display_results(results):
    """Affiche les résultats de manière structurée et claire."""
    if "error" in results:
        st.error(f"Erreur lors de l'analyse: {results['error']}")
        if "raw_response" in results:
            st.text_area("Réponse brute:", results["raw_response"], height=300)
        return
    
    st.markdown("## Résultats de l'analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Informations générales")
        st.info(f"**Type de document:** {results.get('Type de document', 'Non spécifié')}")
        st.info(f"**Période concernée:** {results.get('Année ou période concernée', 'Non spécifiée')}")
        st.info(f"**Montant total des charges:** {results.get('Montant total des charges', 'Non spécifié')}")
    
    with col2:
        st.markdown("### Informations financières")
        st.info(f"**Quote-part du locataire:** {results.get('Quote-part du locataire', 'Non spécifiée')}")
        st.info(f"**Montant facturé au locataire:** {results.get('Montant facturé au locataire', 'Non spécifié')}")
        st.info(f"**Provisions déjà versées:** {results.get('Montant des provisions déjà versées', 'Non spécifié')}")
        st.info(f"**Solde:** {results.get('Solde', 'Non spécifié')}")
    
    # Afficher la répartition des charges si disponible
    if "Répartition des charges par poste" in results and results["Répartition des charges par poste"]:
        st.markdown("### Répartition des charges par poste")
        
        # Convertir en dictionnaire si ce n'est pas déjà le cas
        repartition = results["Répartition des charges par poste"]
        if isinstance(repartition, str):
            try:
                repartition = json.loads(repartition)
            except:
                pass
        
        if isinstance(repartition, dict):
            # Créer un DataFrame pour l'affichage
            df = pd.DataFrame({
                'Poste': list(repartition.keys()),
                'Montant': list(repartition.values())
            })
            
            # Nettoyer les montants pour le graphique
            df['Montant_Numeric'] = df['Montant'].apply(
                lambda x: float(str(x).replace('€', '').replace(',', '.').replace(' ', ''))
                if isinstance(x, str) else float(x)
            )
            
            # Trier par montant décroissant
            df = df.sort_values('Montant_Numeric', ascending=False)
            
            # Afficher le tableau
            st.dataframe(df[['Poste', 'Montant']])
            
            # Créer un graphique
            chart_df = df.copy()
            chart_df = chart_df.set_index('Poste')
            st.bar_chart(chart_df[['Montant_Numeric']])
        else:
            st.write(repartition)
    
    # Bouton de téléchargement des résultats
    st.markdown(download_json(results), unsafe_allow_html=True)
    
    # Afficher les résultats JSON bruts dans un expander
    with st.expander("Voir les résultats bruts (JSON)"):
        st.json(results) st.columns(2)
    
    with col1:
        st.markdown("### Informations générales")
        st.info(f"**Type de document:** {results.get('Type de document', 'Non spécifié')}")
        st.info(f"**Période concernée:** {results.get('Année ou période concernée', 'Non spécifiée')}")
        st.info(f"**Montant total des charges:** {results.get('Montant total des charges', 'Non spécifié')}")
    
    with col2:
        st.markdown("### Informations financières")
        st.info(f"**Quote-part du locataire:** {results.get('Quote-part du locataire', 'Non spécifiée')}")
        st.info(f"**Montant facturé au locataire:** {results.get('Montant facturé au locataire', 'Non spécifié')}")
        st.info(f"**Provisions déjà versées:** {results.get('Montant des provisions déjà versées', 'Non spécifié')}")
        st.info(f"**Solde:** {results.get('Solde', 'Non spécifié')}")
    
    # Afficher la répartition des charges si disponible
    if "Répartition des charges par poste" in results and results["Répartition des charges par poste"]:
        st.markdown("### Répartition des charges par poste")
        
        # Convertir en dictionnaire si ce n'est pas déjà le cas
        repartition = results["Répartition des charges par poste"]
        if isinstance(repartition, str):
            try:
                repartition = json.loads(repartition)
            except:
                pass
        
        if isinstance(repartition, dict):
            # Créer un DataFrame pour l'affichage
            df = pd.DataFrame({
                'Poste': list(repartition.keys()),
                'Montant': list(repartition.values())
            })
            
            # Nettoyer les montants pour le graphique
            df['Montant_Numeric'] = df['Montant'].apply(
                lambda x: float(str(x).replace('€', '').replace(',', '.').replace(' ', ''))
                if isinstance(x, str) else float(x)
            )
            
            # Trier par montant décroissant
            df = df.sort_values('Montant_Numeric', ascending=False)
            
            # Afficher le tableau
            st.dataframe(df[['Poste', 'Montant']])
            
            # Créer un graphique
            chart_df = df.copy()
            chart_df = chart_df.set_index('Poste')
            st.bar_chart(chart_df[['Montant_Numeric']])
        else:
            st.write(repartition)
    
    # Bouton de téléchargement des résultats
    st.markdown(download_json(results), unsafe_allow_html=True)
    
    # Afficher les résultats JSON bruts dans un expander
    with st.expander("Voir les résultats bruts (JSON)"):
        st.json(results) st.columns(2)
    
    with col1:
        st.markdown("### Informations générales")
        st.info(f"**Type de document:** {results.get('Type de document', 'Non spécifié')}")
        st.info(f"**Période concernée:** {results.get('Année ou période concernée', 'Non spécifiée')}")
        st.info(f"**Montant total des charges:** {results.get('Montant total des charges', 'Non spécifié')}")
    
    with col2:
        st.markdown("### Informations financières")
        st.info(f"**Quote-part du locataire:** {results.get('Quote-part du locataire', 'Non spécifiée')}")
        st.info(f"**Montant facturé au locataire:** {results.get('Montant facturé au locataire', 'Non spécifié')}")
        st.info(f"**Provisions déjà versées:** {results.get('Montant des provisions déjà versées', 'Non spécifié')}")
        st.info(f"**Solde:** {results.get('Solde', 'Non spécifié')}")
    
    # Afficher la répartition des charges si disponible
    if "Répartition des charges par poste" in results and results["Répartition des charges par poste"]:
        st.markdown("### Répartition des charges par poste")
        
        # Convertir en dictionnaire si ce n'est pas déjà le cas
        repartition = results["Répartition des charges par poste"]
        if isinstance(repartition, str):
            try:
                repartition = json.loads(repartition)
            except:
                pass
        
        if isinstance(repartition, dict):
            # Créer un DataFrame pour l'affichage
            df = pd.DataFrame({
                'Poste': list(repartition.keys()),
                'Montant': list(repartition.values())
            })
            
            # Afficher le tableau
            st.dataframe(df)
            
            # Créer un graphique
            st.bar_chart(df.set_index('Poste'))
        else:
            st.write(repartition)
    
    # Bouton de téléchargement des résultats
    st.markdown(download_json(results), unsafe_allow_html=True)
    
    # Afficher les résultats JSON bruts dans un expander
    with st.expander("Voir les résultats bruts (JSON)"):
        st.json(results)

def main():
    """Fonction principale de l'application Streamlit."""
    st.set_page_config(
        page_title="Analyseur de Charges - Baux Commerciaux",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Analyseur de Tableaux de Charges pour Baux Commerciaux")
    st.markdown("""
    Cette application analyse vos tableaux de charges et redditions de charges pour des baux commerciaux.
    Téléchargez un fichier PDF, Excel ou CSV et obtenez une analyse détaillée grâce à l'OCR et à l'IA.
    """)
    
    # Sidebar pour la configuration
    st.sidebar.title("Configuration")
    
    # Récupération des clés API depuis les secrets de Streamlit
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.success("✅ Clé API OpenAI chargée depuis les secrets")
    except Exception as e:
        st.sidebar.error("❌ Impossible de charger la clé API OpenAI depuis les secrets")
        api_key = st.sidebar.text_input("Clé API OpenAI (secours)", type="password")
        
    # Récupération de la clé OCR API si disponible
    try:
        ocr_api_key = st.secrets["OCR_API_KEY"]
        st.sidebar.success("✅ Clé API OCR chargée depuis les secrets")
    except Exception as e:
        ocr_api_key = None
    
    # Paramètres d'analyse avancés
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres d'analyse")
    
    use_ai = st.sidebar.checkbox("Utiliser l'IA pour l'analyse", value=True, 
                              help="Désactivez pour utiliser uniquement l'extraction directe (plus rapide mais moins précis)")
    
    ocr_quality = st.sidebar.select_slider(
        "Qualité OCR", 
        options=["Rapide", "Standard", "Précise"],
        value="Standard",
        help="Plus la qualité est élevée, plus l'analyse sera précise mais plus le temps de traitement sera long"
    )
    
    # Ajouter un expander pour les informations techniques
    with st.sidebar.expander("Informations techniques"):
        st.markdown("""
        - **Extraction directe** : analyse le document directement en utilisant des expressions régulières
        - **Analyse IA** : utilise GPT-4o-mini pour une analyse plus approfondie
        - **Qualité OCR** : affecte la précision de la reconnaissance de texte dans les PDFs
        """)
    
    # Informations supplémentaires dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ## À propos
    Cette application utilise:
    - OCR (Reconnaissance Optique de Caractères) pour extraire le texte des PDFs
    - GPT-4o-mini pour analyser le contenu
    - Streamlit pour l'interface utilisateur
    
    ## Formats supportés
    - PDF (avec extraction OCR)
    - Excel (.xlsx, .xls)
    - CSV
    """)
    
    # Onglets principaux
    tab1, tab2 = st.tabs(["Analyse simple", "Analyse comparative"])
    
    with tab1:
        # Zone principale pour l'analyse simple
        st.markdown("## Téléchargement du fichier")
        uploaded_file = st.file_uploader("Choisissez un fichier à analyser", type=["pdf", "xlsx", "xls", "csv"], key="single_file")
        
        if uploaded_file is not None:
            st.markdown("---")
            
            # Afficher des informations sur le fichier téléchargé
            file_details = {
                "Nom du fichier": uploaded_file.name,
                "Type de fichier": uploaded_file.type,
                "Taille": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            st.markdown("### Détails du fichier")
            for key, value in file_details.items():
                st.info(f"**{key}:** {value}")
            
            # Bouton pour lancer l'analyse
            if st.button("Analyser le document"):
                if not api_key:
                    st.error("Aucune clé API OpenAI disponible. Vérifiez les secrets Streamlit ou entrez une clé manuellement.")
                else:
                    try:
                        # Afficher un spinner pendant l'analyse
                        with st.spinner("Analyse en cours, veuillez patienter..."):
                            # Configurer les options d'OCR selon le choix de qualité
                            ocr_config = {
                                "Rapide": "--oem 0 --psm 6",
                                "Standard": "--oem 3 --psm 6",
                                "Précise": "--oem 3 --psm 11 -l fra"
                            }
                            
                            # Initialiser l'analyseur avec les paramètres configurés
                            analyzer = ChargesAnalyzer(
                                api_key=api_key, 
                                ocr_api_key=ocr_api_key, 
                                use_ai=use_ai,
                                tesseract_config=ocr_config[ocr_quality]
                            )
                            
                            # Traiter le fichier
                            results = analyzer.process_file(uploaded_file)
                            
                            # Afficher les résultats
                            display_results(results)
                    except Exception as e:
                        st.error(f"Une erreur est survenue: {str(e)}")
                        st.exception(e)  # Affiche la trace complète de l'erreur pour le débogage
    
    with tab2:
        # Zone pour l'analyse comparative
        st.markdown("## Comparaison de documents")
        st.markdown("""
        Cette fonctionnalité vous permet de comparer deux documents de charges pour analyser les différences 
        (par exemple, comparer deux années consécutives ou deux locataires différents).
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Premier document")
            uploaded_file1 = st.file_uploader("Choisissez le premier fichier", type=["pdf", "xlsx", "xls", "csv"], key="file1")
        
        with col2:
            st.markdown("### Second document")
            uploaded_file2 = st.file_uploader("Choisissez le second fichier", type=["pdf", "xlsx", "xls", "csv"], key="file2")
        
        if uploaded_file1 is not None and uploaded_file2 is not None:
            if st.button("Comparer les documents"):
                if not api_key:
                    st.error("Aucune clé API OpenAI disponible. Vérifiez les secrets Streamlit ou entrez une clé manuellement.")
                else:
                    try:
                        # Analyser le premier document
                        with st.spinner("Analyse du premier document en cours..."):
                            analyzer = ChargesAnalyzer(
                                api_key=api_key, 
                                ocr_api_key=ocr_api_key,
                                use_ai=use_ai,
                                tesseract_config=ocr_config[ocr_quality]
                            )
                            results1 = analyzer.process_file(uploaded_file1)
                            
                        # Analyser le second document
                        with st.spinner("Analyse du second document en cours..."):
                            results2 = analyzer.process_file(uploaded_file2)
                        
                        # Afficher une comparaison des résultats
                        st.markdown("## Résultats de la comparaison")
                        
                        # Comparer les informations générales
                        compare_cols = st.columns(2)
                        with compare_cols[0]:
                            st.markdown(f"### {uploaded_file1.name}")
                            st.info(f"**Type de document:** {results1.get('Type de document', 'Non spécifié')}")
                            st.info(f"**Période concernée:** {results1.get('Année ou période concernée', 'Non spécifiée')}")
                            st.info(f"**Montant total des charges:** {results1.get('Montant total des charges', 'Non spécifié')}")
                        
                        with compare_cols[1]:
                            st.markdown(f"### {uploaded_file2.name}")
                            st.info(f"**Type de document:** {results2.get('Type de document', 'Non spécifié')}")
                            st.info(f"**Période concernée:** {results2.get('Année ou période concernée', 'Non spécifiée')}")
                            st.info(f"**Montant total des charges:** {results2.get('Montant total des charges', 'Non spécifié')}")
                        
                        # Comparer les répartitions de charges
                        if ("Répartition des charges par poste" in results1 and 
                            "Répartition des charges par poste" in results2):
                            
                            st.markdown("### Comparaison des postes de charges")
                            
                            repartition1 = results1["Répartition des charges par poste"]
                            repartition2 = results2["Répartition des charges par poste"]
                            
                            # Convertir en dictionnaires si nécessaire
                            if isinstance(repartition1, str):
                                try: repartition1 = json.loads(repartition1)
                                except: pass
                            if isinstance(repartition2, str):
                                try: repartition2 = json.loads(repartition2)
                                except: pass
                            
                            if isinstance(repartition1, dict) and isinstance(repartition2, dict):
                                # Créer un DataFrame pour la comparaison
                                all_postes = list(set(list(repartition1.keys()) + list(repartition2.keys())))
                                
                                comparison_data = {
                                    'Poste': all_postes,
                                    'Document 1': [repartition1.get(poste, "N/A") for poste in all_postes],
                                    'Document 2': [repartition2.get(poste, "N/A") for poste in all_postes],
                                }
                                
                                df_comp = pd.DataFrame(comparison_data)
                                
                                # Convertir en valeurs numériques pour le calcul de la variation
                                def convert_to_numeric(val):
                                    if val == "N/A": 
                                        return float('nan')
                                    return float(str(val).replace('€', '').replace(',', '.').replace(' ', ''))
                                
                                df_comp['Valeur 1'] = df_comp['Document 1'].apply(convert_to_numeric)
                                df_comp['Valeur 2'] = df_comp['Document 2'].apply(convert_to_numeric)
                                
                                # Calculer la variation
                                df_comp['Variation'] = ((df_comp['Valeur 2'] - df_comp['Valeur 1']) / df_comp['Valeur 1'] * 100).round(2)
                                df_comp['Variation'] = df_comp['Variation'].apply(lambda x: f"{x}%" if not pd.isna(x) else "N/A")
                                
                                # Afficher le tableau de comparaison
                                st.dataframe(df_comp[['Poste', 'Document 1', 'Document 2', 'Variation']])
                    
                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la comparaison: {str(e)}")
                        st.exception(e)

if __name__ == "__main__":
    # Afficher les informations sur les secrets disponibles (sans révéler les valeurs)
    st.sidebar.markdown("### État des secrets")
    secrets_status = {
        "OPENAI_API_KEY": "OPENAI_API_KEY" in st.secrets if hasattr(st, "secrets") else False,
        "OCR_API_KEY": "OCR_API_KEY" in st.secrets if hasattr(st, "secrets") else False
    }
    
    for secret_name, is_available in secrets_status.items():
        if is_available:
            st.sidebar.success(f"✅ {secret_name} configuré")
        else:
            st.sidebar.warning(f"⚠️ {secret_name} non configuré")
    
    # Pour les erreurs liées à l'OCR sur Streamlit Cloud
    try:
        import pytesseract
    except ImportError:
        st.error("La bibliothèque pytesseract n'est pas installée. Ajoutez-la à requirements.txt")
    
    try:
        # Tester si tesseract est installé
        pytesseract.get_tesseract_version()
    except Exception as e:
        st.warning(f"Tesseract OCR n'est pas correctement configuré: {str(e)}")
        st.info("Sur Streamlit Cloud, ajoutez 'tesseract-ocr' à votre packages.txt")
    
    main()
