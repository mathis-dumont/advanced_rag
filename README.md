# Assistant Technique RAG avec Interface Streamlit

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)

## Description du Projet

Ce projet implémente un système de Questions-Réponses (QA) de bout en bout, basé sur l'architecture **RAG (Retrieval Augmented Generation)**. Il est conçu pour servir d'assistant technique intelligent pour des agents de maintenance, en répondant à leurs questions en se basant **exclusivement** sur une base de documents techniques fournie (fichiers `.doc` et `.docx`).

Le pipeline complet gère la conversion des documents, l'extraction de texte et la description d'images (approche multimodale), la segmentation sémantique du contenu, la génération d'embeddings vectoriels, l'indexation dans une base de données **FAISS**, et la génération de réponses contextuelles et sourcées via un grand modèle de langage **Mistral**.

Le projet propose à la fois une interface en ligne de commande (CLI) pour l'administration et une interface web interactive développée avec **Streamlit**.

## Table des Matières

1.  [Description du Projet](#description-du-projet)
2.  [Architecture et Flux de Données](#architecture-et-flux-de-données)
3.  [Fonctionnalités Clés](#fonctionnalités-clés)
4.  [Structure du Projet](#structure-du-projet)
5.  [Installation et Configuration](#installation-et-configuration)
6.  [Utilisation](#utilisation)
7.  [Description Détaillée des Modules](#description-détaillée-des-modules)
8.  [Concepts RAG Avancés Implémentés](#concepts-rag-avancés-implémentés)
9.  [Pistes d'Amélioration](#pistes-damélioration)

## Architecture et Flux de Données

Le système suit un pipeline RAG classique en plusieurs étapes :

1.  **Ingestion et Prétraitement** (`loaders.py`, `converters.py`): Les documents Word (`.doc`, `.docx`) sont convertis en PDF via LibreOffice. Le texte et les images sont extraits de ces PDF. Un modèle multimodal Mistral génère des descriptions pour les images, qui sont ensuite réinjectées dans le flux textuel pour un contexte enrichi.

2.  **Segmentation (Chunking)** (`chunking.py`): Le contenu textuel de chaque page est intelligemment divisé en segments (chunks) de taille contrôlée, en utilisant `spaCy` pour respecter les frontières de phrases et `tiktoken` pour un comptage précis des tokens. Un chevauchement (overlap) préserve le contexte entre les chunks.

3.  **Vectorisation (Embedding)** (`embedding.py`): Chaque chunk est transformé en un vecteur numérique (embedding) via le modèle `mistral-embed`. Ces vecteurs capturent la signification sémantique du texte.

4.  **Indexation** (`index_manager.py`): Les embeddings sont stockés et indexés dans une base de données vectorielle **FAISS** (optimisée pour la recherche par similarité cosinus) pour une récupération ultra-rapide.

5.  **Récupération (Retrieval)** (`retriever.py`): Lorsqu'un utilisateur pose une question, celle-ci est vectorisée. Le retriever recherche dans l'index FAISS les chunks les plus sémantiquement similaires. Une "fenêtre" de chunks adjacents est également récupérée pour élargir le contexte.

6.  **Génération** (`pipeline.py`): Les chunks récupérés (le contexte) et la question sont assemblés dans un prompt structuré. Ce prompt est envoyé à un LLM (ex: `mistral-medium`) qui génère une réponse en se basant uniquement sur le contexte fourni.

7.  **Présentation** (`streamlit_chat.py`, `app.py`): La réponse, enrichie de ses sources, est présentée à l'utilisateur via une interface de chat Streamlit ou en ligne de commande.

## Fonctionnalités Clés

*   **Conversion Automatique** des documents `.doc` et `.docx` en PDF.
*   **Enrichissement Multimodal** avec description automatique des images.
*   **Chunking Sémantique** intelligent avec respect des phrases et contrôle de la taille des tokens.
*   **Indexation Vectorielle Performante** avec FAISS.
*   **Récupération Contextuelle Dynamique** via une "fenêtre" de chunks adjacents.
*   **Génération de Réponses Sourcées** avec citation automatique des documents et pages.
*   **Gestion Flexible de la Base de Données** (`auto`, `rebuild`, `incremental`).
*   **Double Interface** : CLI complète pour l'administration et interface web interactive avec Streamlit.
*   **Prompt Engineering Avancé** pour des réponses techniques précises et factuelles.

## Structure du Projet

Le projet suit une structure `src-layout` pour une séparation nette entre le code source et les autres fichiers.

```

├── data/
│   ├── input/                # Placez vos documents .doc/.docx ici
│   ├── static/                   # Fichiers PDF générés pour l'accès web
│   └── vector_store/         # Base de données vectorielle (créée automatiquement)
├── src/
│   ├── rag_agent/            # Paquet principal de l'application RAG
│   │   ├── components/       # Modules RAG (chunker, retriever...)
│   │   ├── io/               # Modules d'entrée/sortie (loaders, converters)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── pipeline.py
│   ├── app.py                # Point d'entrée de la CLI
│   └── streamlit_chat.py     # Point d'entrée de l'UI Streamlit
├── .env                      # Fichier pour les secrets (clé API)
├── .gitignore
├── README.md
├── requirements.txt
└── settings.json             # Fichier de configuration principal
```

## Installation et Configuration

Suivez ces étapes pour mettre en place l'environnement.

### Prérequis

1.  **Python** (version 3.10+ recommandée).
2.  **LibreOffice** : Doit être installé sur le système.
    *   Sur Linux (Debian/Ubuntu) : `sudo apt-get install libreoffice`
    *   Sur macOS/Windows : Installez depuis le site officiel et assurez-vous que l'exécutable `soffice` est dans le PATH de votre système.

### Étapes d'Installation

1.  **Cloner le dépôt**
    ```bash
    git clone <url_du_depot>
    cd DID.SIMO.AI_AGENT_EDITOR
    ```

2.  **Créer un environnement virtuel et l'activer**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installer les dépendances Python**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Télécharger le modèle linguistique spaCy**
    ```bash
    python -m spacy download fr_core_news_md
    ```

5.  **Configurer la Clé API Mistral**
    Créez un fichier `.env` à la racine du projet et ajoutez votre clé :
    ```env
    MISTRAL_API_KEY="VOTRE_CLE_API_MISTRAL_ICI"
    ```

6.  **Configurer le Projet**
    Vérifiez et adaptez si nécessaire le fichier `settings.json` à la racine. La configuration par défaut est conçue pour la structure actuelle du projet.
    ```json
    {
      "data_dir": "data/input",
      "pdf_output_dir": "static",
      "db_dir": "data/vector_store",
      "nlp_model": "fr_core_news_md",
      "tokenizer_encoding": "cl100k_base",
      "embedding_model": "mistral-embed",
      "chat_model": "mistral-medium-2505",
      "chunk_max_tokens": 350,
      "chunk_overlap": 40,
      "k": 10,
      "window": 1,
      "process_images": true
    }
    ```

## Utilisation

### Étape 1 : Ajouter des Documents
Placez vos fichiers `.doc` et `.docx` dans le dossier `data/input/`.

### Étape 2 : Lancer l'Interface Web (Streamlit)

Pour une expérience stable, lancez toujours l'application avec la commande suivante, qui désactive le "File Watcher" problématique :

```bash
streamlit run src/streamlit_chat.py --server.fileWatcherType none
```

Lors du premier lancement, l'application va automatiquement :
-   Convertir vos documents Word en PDF dans le dossier `static/`.
-   Construire la base de données vectorielle dans `data/vector_store/`.
-   Cela peut prendre plusieurs minutes en fonction du volume de documents.

### Étape 3 : Utiliser l'Interface en Ligne de Commande (CLI)
L'interface en ligne de commande est idéale pour l'administration et les tests.

*   **Mode interactif simple :**
    ```bash
    python src/app.py
    ```
*   **Forcer la reconstruction de la base :**
    ```bash
    python src/app.py --mode rebuild
    ```
*   **Mettre à jour la base avec de nouveaux fichiers :**
    ```bash
    python src/app.py --mode incremental
    ```
*   **Poser une question directement :**
    ```bash
    python src/app.py --query "Comment accéder physiquement aux serveurs PDA et WINCC ?"
    ```

## Description Détaillée des Modules
*(Cette section est basée sur l'excellent travail de votre README original, mise à jour pour refléter l'architecture finale)*

-   **`src/app.py`** : Point d'entrée de la CLI. Gère les arguments, charge la configuration depuis `settings.json` et orchestre le `RAGPipeline`.
-   **`src/streamlit_chat.py`** : Point d'entrée de l'UI web. Gère la logique d'affichage, la mise en cache des composants, et l'interaction avec le `RAGPipeline`. Utilise `st.session_state` pour gérer les objets lourds (comme le `Chunker`) et éviter les conflits de framework.
-   **`src/rag_agent/pipeline.py`** : Cœur de l'orchestration. La classe `RAGPipeline` connecte tous les composants. Elle gère la logique de construction de la base, la génération des prompts, et la formulation des réponses finales.
-   **`src/rag_agent/io/converters.py`** : Gère la conversion de documents Word en PDF via un appel `subprocess` à LibreOffice (`soffice`).
-   **`src/rag_agent/io/loaders.py`** : Orchestre le chargement des fichiers, leur conversion, et l'extraction de contenu multimodal (texte + descriptions d'images).
-   **`src/rag_agent/components/chunking.py`** : Contient la classe `Chunker` qui segmente le texte en morceaux sémantiquement cohérents et de taille contrôlée.
-   **`src/rag_agent/components/embedding.py`** : Gère la communication avec l'API Mistral pour transformer les chunks de texte en vecteurs.
-   **`src/rag_agent/components/index_manager.py`** : Encapsule toute la logique de création, sauvegarde et chargement de l'index vectoriel FAISS et des métadonnées associées.
-   **`src/rag_agent/components/retriever.py`** : Prend une question, la vectorise et interroge l'index FAISS pour trouver les chunks les plus pertinents.

## Concepts RAG Avancés Implémentés
*(Cette section est conservée de votre README original car elle est excellente)*

Ce projet met en œuvre plusieurs techniques avancées pour améliorer la qualité et la pertinence du système RAG :

1.  **Chunking contextuel avec chevauchement**
2.  **Gestion multiformat et conversion**
3.  **Extraction et description d'images (Multimodalité)**
4.  **Indexation vectorielle optimisée** (FAISS avec `IndexFlatIP` et normalisation L2)
5.  **Embeddings de pointe** (`mistral-embed`)
6.  **Fenêtre contextuelle dynamique** pour enrichir le contexte du LLM.
7.  **Mise à jour incrémentale** de la base de connaissances.
8.  **Prompt engineering avancé** pour guider le LLM vers des réponses factuelles et techniques.
9.  **Traçabilité des sources** avec liens cliquables vers les PDF.

## Pistes d'Amélioration

1.  **Re-ranking des Chunks** : Intégrer un modèle "cross-encoder" après le retrieval initial pour affiner la pertinence des chunks passés au LLM.
2.  **Reformulation de la Question** : Utiliser un LLM pour améliorer ou décomposer la question de l'utilisateur avant la recherche vectorielle.
3.  **Gestion Avancée des Métadonnées** : Extraire des métadonnées (titres, sections, dates) pour permettre un filtrage pendant la recherche.
4.  **Évaluation Rigoureuse** : Mettre en place un framework d'évaluation (ex: Ragas) pour mesurer objectivement la performance du système.
5.  **Passage à l'échelle** : Pour des corpus de documents beaucoup plus importants (> 1 Go), remplacer `IndexFlatIP` par des index plus scalables comme `IndexIVF` ou `HNSW`.
