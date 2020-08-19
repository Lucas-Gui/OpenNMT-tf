#Extraction et injection d'attention
Ce code est une partie de mon travail en stage en été 2020.

Il a pour but de :
1. Caractériser les têtes à autoattention d'un Transformer selon leur capacité à identifier 
des dépendances grammaticales, de la même façon que [https://arxiv.org/pdf/1905.09418.pdf](Voita et al.)
2. Injecter des valeurs pour l'attention entre deux tokens choisis sur un ensembre de têtes données,
pour modifier ou ajouter une information grammaticale au modèle.

Contact : lucas.guirardel@polytechnique.edu

### Remarques générales:

Tout est conçu pour la traduction français -> allemand. 
Pour adapter à d’autres paires de langues, il faudra ajuster en fonction du tagging scheme de Spacy pour la langue voulue, ou prendre un autre outil (Stanford Dependencies ?) 

### Librairies nécessaires:

- Fasttext 
- Tqdm  
- Spacy avec les modèles voulus (ici, fr_core_news_md, de_core_news_md) 
- Versions modifiées d’OpenNMT 2.11.1 (inclues dans le dossier : 
OpenNMT-tf-ex pour l'extraction de l'attention, 
OpenNMT-tf-inj pour l'injection d'attention)
- Evidemment Tensorflow 
- Matplotlib pour process_result_attention 

## Mesure de l’attention : processus pour un texte sans annotations syntactiques

- Nettoyer le texte brut  
    - Enlever les phrases d’autres langues avec find_fr.py  
    - Enlever les caractères unicode spéciaux (ex : espaces insécables, …) 
    (pas nécessaire mais facilite le travail de spacy) 
        - Remove_spaces.py fait le plus important : remplacer les espaces inhabituels 

- Préparer les annotations syntactiques: make_UD.py  
    - Peut être utilisé pour séparer le corpus en phrases (flag –S) 
    - Préciser les catégories d’intérêt : e.g. --cats gen nb pour avoir l’information de genre et de nombre des mots concernés pour chaque phrase  

- Tokeniser (à voir selon le modèle).  
    - Tokeniser le fichier .sentences créé par make_UD.py  le cas échéant 
    - Eventuellement filtrer les phrases trop longues (par exemple avec reduce.py) (mais ne devrait pas être nécessaire si on se limite bien à des phrases) 

- Aligner tokens et mots : align_pos.py  
    - Donner en paramètres les fichiers précédemment créés par make_UD.py et le fichier de tokens 
    - Etudier la fréquence des différents tokens : rarity.py .Nécessaire pour étudier la focalisation des têtes sur les tokens plus rares 

- Etudier l’attention : analyze_attn_dep.py  
    - Donner en paramètres les fichiers de tokens, de heads, de dependecies, et des catégories étudiées alignées par align_pos.py. 
    - Paramètre --dep : analyse l’attention sur ces dépendances (par ex : nsubj obj amod advmod pour reproduire les expériences de Voita) 
    - Paramètre -R : fichier de rareté des tokens produit par rarity.py 
    - Le script crée un objet AttnResult pickled.  

- Afficher les résultats: process_results_attn.py  
    - Prend en entrée un objet AttnResult picklé 
    - Crée différents graphiques, ainsi qu’un fichier score.txt qui indique les meilleures têtes pour chaque dépendance. 
    - Est à modifier manuellement... 

 
## Injection d’attention

- Mesures des statistiques sur l’attention: measure_attn.py : 
    - Calcule des statistiques (moyenne, écart-type, premier et dernier décile) des valeurs de l’attention pour chaque tête du modèle 
    - Produit un dossier contenant des array numpy 
    - Attention, très gourmand en mémoire (doit garder en mémoire toutes les valeurs de l'attention pour le calcul des quantiles)
    Il est recommandé de ne lui donner qu'un extrait du corpus.

- Inject_tasks.py  
    - Utilise les dépendances, heads et labels alignées par align_pos.py. Utilise le dossier produit par measure_attn.py 
    - Peut effectuer un certain nombre de tâches :  
        - injection d'un bruit aléatoire sur l'attention du verbe vers le sujet 
        - diminution de l'attention du sujet vers le verbe et du verbe vers le sujet 
        - "échange" de l'attention entre deux paires sujet-verbe 
        - augmentation de l'attention verbe-sujet sur des verbes à l'actif, sur une tête s'activant sur cette relation pour des verbes au passif 
        - "échange" de l'attention entre sujet et objet d'un même verbe 
        - ... 
        - Aisé d’ajouter de nouvelles tâches dans le code 

    - Très lent, car ne fait pas de batch pour l’inférence. 

- Evaluer l’impact des différentes tâches avec eval_inject.py (assez rudimentaire): 
    - Compte et extrait les lignes qui ont été modifées 
    - Peut appliquer des tests (pour l’instant, se limite à : est-ce que les paires (heads/tail) ont été modifiées pour une dépendance donnée 
    - Sinon, à faire à la main.

## Versions modifiées d'OpenNMT
Les modifications que j'ai apportées sont marquées par "#<mod>" dans le code.

### OpenNMT-tf-ex
Version modifiée d'OpenNMT-tf pour pouvoir obtenir les valeurs de l'attention calculées 
par le modèle lors de l'inférence.
On s'intéresse seulement au décodeur d'un GPT2, ou à l'encodeur d'un Transformer.

#### Usage :
Une fois le modèle chargé :
```python 
tokens, attn = model(data, return_attn = True, training = False)
```
`attn` est une numpy.ndarray de taille (L,H,S,S) où L et H indiquent la couche et le numéo de la
tête dans la couche, et S la taille de la phrase.
 `attn[l,h,i,j]` correspond à l'attention du token i vers le mot j.
 
 Remarque : le modèle procède uniquement à l'encodage, pas au décodage. Les tokens
 renvoyés sont ceux de l'entrée.

### OpenNMT-tf-inj:
Version modifiée d'OpenNMT-tf pour pouvoir remplacer manuellement les valeurs de l'attention
entre certains tokens dans certaines têtes, dans un modèle de type Transformer.

#### Usage:
Soit `inj_val` une `np.ndarray` de taille (L,H,S,S) contenant les valeurs à injecter, et
`inj_mask` une `np.ndarray` de même taille, contenant `True` là où les valeurs de l'attention
doivent être modifiées, et `False` ailleurs.

Alors :
```python
_, predict = model(data, training=False, inject=(inj_val, inj_mask))
```
`predict` est le résultat modifié. 

