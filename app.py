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

# Configuration du logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("charges_analyzer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("charges_analyzer")

# CSS personnalisé
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class ChargesAnalyzer:
    """
    Classe pour analyser les tableaux de charges des baux commerciaux 
    à partir de différents formats (Excel, CSV, PDF) en utilisant l'OCR et GPT-4o-mini.
    """
    
    def __init__(self, api_key=None):
        """
        Initialise l'analyseur de charges.
        
        Args:
            api_key (str, optional): Clé API OpenAI.
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Clé API OpenAI non fournie.")
            
        # Initialiser le client OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        # Configuration de Tesseract OCR
        self.tesseract_config = r'--oem 3 --psm 6'
    
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
        Analyse un fichier PDF en utilisant l'OCR et l'extraction directe ou GPT-4o-mini.
        
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
                st.info("Assurez-vous que 'poppler-utils' est installé sur votre serveur Streamlit Cloud. Ajoutez les dépendances dans packages.txt")
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
        
            # D'abord essayer l'extraction directe
            direct_result = self._extract_charges_data(extracted_text)
            
            # Vérifier si l'extraction directe a donné des résultats satisfaisants
            if self._is_extraction_complete(direct_result):
                st.success("Analyse réussie avec l'extraction directe!")
                return direct_result
            else:
                # Sinon, utiliser GPT-4o-mini
                st.info("Utilisation de l'IA pour compléter l'analyse...")
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
            import openpyxl
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
            
            # D'abord essayer l'extraction directe
            direct_result = self._extract_charges_data(combined_text)
            
            # Vérifier si l'extraction directe a donné des résultats satisfaisants
            if self._is_extraction_complete(direct_result):
                st.success("Analyse réussie avec l'extraction directe!")
                return direct_result
            else:
                # Sinon, utiliser GPT-4o-mini
                st.info("Utilisation de l'IA pour compléter l'analyse...")
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
            
            # D'abord essayer l'extraction directe
            direct_result = self._extract_charges_data(csv_text)
            
            # Vérifier si l'extraction directe a donné des résultats satisfaisants
            if self._is_extraction_complete(direct_result):
                st.success("Analyse réussie avec l'extraction directe!")
                return direct_result
            else:
                # Sinon, utiliser GPT-4o-mini
                st.info("Utilisation de l'IA pour compléter l'analyse...")
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
        
        IMPORTANT: Réponds uniquement avec le JSON structuré des résultats, sans aucun texte d'introduction ni formatage supplémentaire comme des backticks.
        Ne pas inclure "```json" ni "```" dans ta réponse. Renvoie uniquement le JSON pur.
        """
        
        try:
            # Appeler l'API OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé dans l'analyse de documents financiers immobiliers. Tu réponds uniquement en JSON valide sans formatage supplémentaire."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Extraire la réponse
            analysis_text = response.choices[0].message.content
            
            # Tenter d'extraire le JSON avec notre méthode sécurisée
            analysis_result = self._extract_json_safely(analysis_text)
            
            if analysis_result:
                st.success("Analyse réussie avec l'IA!")
                return analysis_result
            else:
                st.error("Impossible d'extraire un JSON valide de la réponse")
                st.text_area("Réponse brute:", analysis_text, height=300)
                return {"error": "Format de réponse invalide", "raw_response": analysis_text}
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")
            return {"error": str(e)}
    
    def _extract_json_safely(self, text):
        """
        Tente plusieurs approches pour extraire un JSON valide d'un texte.
        
        Args:
            text (str): Texte contenant potentiellement du JSON.
            
        Returns:
            dict: Le dictionnaire JSON extrait ou None si l'extraction échoue.
        """
        # Supprimer tous les backticks et mentions "json"
        cleaned_text = text.replace('```json', '').replace('```', '').replace('json', '').strip()
        
        # Approche 1: Essayer de charger directement le texte nettoyé
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
        
        # Approche 2: Chercher les délimiteurs {} les plus externes
        try:
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = cleaned_text[start_idx:end_idx]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Approche 3: Correction des problèmes de guillemets potentiels
        try:
            # Remplacer les guillemets simples par des guillemets doubles
            fixed_text = cleaned_text.replace("'", '"')
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Approche 4: Utiliser une expression régulière pour extraire le JSON
        try:
            json_pattern = r'(\{.*\})'
            match = re.search(json_pattern, cleaned_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
        
        # Approche 5: Tentative de correction manuelle de formats JSON courants
        try:
            # Correction de certaines erreurs de formatage fréquentes
            corrected_text = cleaned_text
            # Remplacer les virgules à la fin des objets
            corrected_text = re.sub(r',\s*}', '}', corrected_text)
            # Ajouter des guillemets aux clés non quotées
            corrected_text = re.sub(r'(\w+):', r'"\1":', corrected_text)
            return json.loads(corrected_text)
        except json.JSONDecodeError:
            pass
        
        # Si toutes les tentatives échouent
        return None
    
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
            
            # Afficher le tableau
            st.dataframe(df)
        else:
            st.write(repartition)
    
    # Bouton de téléchargement des résultats
    st.markdown(download_json(results), unsafe_allow_html=True)

def main():
    """Fonction principale de l'application Streamlit."""
    st.title("Analyseur de Redditions de Charges")
    st.markdown("""
    Cette application analyse précisément vos tableaux de charges et redditions de charges pour des baux commerciaux.
    Téléchargez un fichier PDF, Excel ou CSV et obtenez une analyse complète.
    """)
    
    # Récupération de la clé API depuis les secrets de Streamlit
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        api_key = st.text_input("Clé API OpenAI", type="password")
    
    # Zone principale
    uploaded_file = st.file_uploader("Choisissez un fichier à analyser", type=["pdf", "xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        # Afficher des informations sur le fichier téléchargé
        st.info(f"Fichier chargé: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Bouton pour lancer l'analyse
        if st.button("Analyser le document"):
            if not api_key:
                st.error("Veuillez fournir une clé API OpenAI.")
            else:
                try:
                    # Afficher un spinner pendant l'analyse
                    with st.spinner("Analyse en cours, veuillez patienter..."):
                        # Initialiser l'analyseur
                        analyzer = ChargesAnalyzer(api_key=api_key)
                        
                        # Traiter le fichier
                        results = analyzer.process_file(uploaded_file)
                        
                        # Afficher les résultats
                        display_results(results)
                except Exception as e:
                    st.error(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    # Vérifier si tesseract est installé
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        st.warning(f"Tesseract OCR n'est pas correctement configuré: {str(e)}")
        st.info("Sur Streamlit Cloud, ajoutez 'tesseract-ocr' à votre packages.txt")
    
    main()
