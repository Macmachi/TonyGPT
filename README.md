# TonyGPT

TonyGPT est un robot humanoïde qui comprend et parle 20 langues avec une connaissance approfondie. Voici une description détaillée de ses fonctionnalités principales :

## Reconnaissance faciale

- Utilise un modèle de détection de visage pré-entraîné (res10_300x300_ssd_iter_140000_fp16.caffemodel) pour détecter les visages dans les images de la caméra.
- Suit le visage détecté en ajustant la position de la tête du robot à l'aide de servomoteurs contrôlés par un algorithme PID.
- Si aucun visage n'est détecté, la tête du robot tourne de droite à gauche pour rechercher des personnes.

## Reconnaissance vocale et interaction avec ChatGPT

- Écoute les commandes vocales de l'utilisateur à l'aide de la bibliothèque speech_recognition.
- Reconnaît la parole en utilisant l'API de reconnaissance vocale de Google.
- Interagit avec l'API GPT-4 d'OpenAI pour générer des réponses aux questions de l'utilisateur.
- Maintient un historique de conversation pour un contexte continu.
- Détecte des mots-clés spécifiques comme "stop", "bonjour", "squat", etc., et déclenche des actions correspondantes.

## Synthèse vocale

- Convertit le texte en parole en utilisant l'API de synthèse vocale d'OpenAI.
- Joue l'audio généré à l'aide de la bibliothèque pygame.
- Fait bouger le robot de manière aléatoire pendant qu'il parle pour une interaction plus naturelle.

## Capture et description d'images

- Capture des images à l'aide de la caméra lorsque l'utilisateur demande au robot de décrire ce qu'il voit.
- Envoie l'image capturée à l'API GPT-4 d'OpenAI pour générer une description textuelle.
- Stocke la description générée dans une mémoire visuelle pour une utilisation ultérieure dans la conversation.

## Mouvements et actions

- Exécute des groupes d'actions prédéfinis (comme s'accroupir, faire des abdos, saluer, etc.) en réponse à des commandes vocales spécifiques.
- Utilise la bibliothèque hiwonder pour contrôler les servomoteurs et exécuter des mouvements fluides.

## Configuration et personnalisation

- Charge les paramètres de configuration à partir de fichiers YAML et INI.
- Permet de personnaliser les paramètres tels que les seuils de confiance, les clés API, etc.

## Future Ideas

- Addition of a servo-motorized arm to extend the robot's physical capabilities.
- Improvement of facial recognition to detect specific faces.
- Integration of new features to make the robot even more versatile.

TonyGPT est un projet open-source qui démontre l'intégration de la vision par ordinateur, du traitement du langage naturel et de la robotique pour créer un compagnon robotique interactif et polyvalent. Le code est modulaire et bien documenté, ce qui permet aux développeurs de l'étendre et de l'adapter à leurs besoins spécifiques.

TonyGPT est un projet open-source qui démontre l'intégration de la vision par ordinateur, du traitement du langage naturel et de la robotique pour créer un compagnon robotique interactif et polyvalent. Le code est modulaire et bien documenté, ce qui permet aux développeurs de l'étendre et de l'adapter à leurs besoins spécifiques.
