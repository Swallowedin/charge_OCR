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
    
    def __init__(self, api_key=None, ocr_api_key=None):
        """
        Initialise l'analyseur de charges.
        
        Args:
            api_key (str, optional): Clé API OpenAI.
            ocr_api_key (str, optional): Clé API OCR (si utilisation d'une API OCR externe).
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Clé API OpenAI non fournie.")
            
        # Initialiser le client OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        # Stocker la clé API OCR si fournie
        self.ocr_api_key = ocr_api_key
        
        # Configuration de Tesseract OCR pour Streamlit Cloud
        # Dans Streamlit Cloud, nous devons utiliser pytesseract avec des configurations spécifiques
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
        st.info(f"Analyse du texte avec GPT-4o-mini en cours...")
        
        # Limiter la taille du texte si nécessaire
        if len(text) > 15000:
            st.warning(f"Le texte a été tronqué de {len(text)} à 15000 caractères")
            text = text[:15000] + "...[tronqué]"
        
        # Définir le prompt pour GPT-4o-mini
        prompt = f"""
        Tu es un expert en analyse de tableaux de charges et redditions de charges pour des baux commerciaux.
        Voici un tableau de charges extrait d'un document {source_type}.
        
        {text}
        
        Analyse ce tableau de charges et extrait les informations suivantes au format JSON :
        1. Type de document (reddition de charges, appel de charges, budget prévisionnel)
        2. Année ou période concernée
        3. Montant total des charges
        4. Répartition des charges par poste (nettoyage, sécurité, maintenance, etc.)
        5. Quote-part du locataire si mentionnée
        6. Montant facturé au locataire
        7. Montant des provisions déjà versées si applicable
        8. Solde (créditeur ou débiteur)
        
        Réponds uniquement avec le JSON structuré des résultats.
        """
        
        try:
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
                    st.success("Analyse réussie!")
                    return analysis_result
                except json.JSONDecodeError as e:
                    st.error(f"Erreur de décodage JSON: {str(e)}")
                    return {"error": "Format de réponse invalide", "raw_response": analysis_text}
            else:
                st.error("Pas de JSON trouvé dans la réponse")
                return {"error": "Pas de JSON trouvé dans la réponse", "raw_response": analysis_text}
            
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API OpenAI: {str(e)}")
            return {"error": str(e)}

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
    
    # Informations supplémentaires dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ## À propos
    Cette application utilise:
    - OCR (Reconnaissance Optique de Caractères) pour extraire le texte des PDFs
    - GPT-4o-mini pour analyser le contenu
    - Streamlit pour l'interface utilisateur
    
    ## Formats supportés
    - PDF
    - Excel (.xlsx, .xls)
    - CSV
    """)
    
    # Zone principale
    st.markdown("## Téléchargement du fichier")
    uploaded_file = st.file_uploader("Choisissez un fichier à analyser", type=["pdf", "xlsx", "xls", "csv"])
    
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
                        # Initialiser l'analyseur avec les deux clés API
                        analyzer = ChargesAnalyzer(api_key=api_key, ocr_api_key=ocr_api_key)
                        
                        # Traiter le fichier
                        results = analyzer.process_file(uploaded_file)
                        
                        # Afficher les résultats
                        display_results(results)
                except Exception as e:
                    st.error(f"Une erreur est survenue: {str(e)}")
                    st.exception(e)  # Affiche la trace complète de l'erreur pour le débogage

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
