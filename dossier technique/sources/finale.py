import pygame
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os.path
import io
from PIL import Image

# Charger les données
donnees = pd.read_csv('athlete_events_FINE.csv')

# Traduire les noms des sports
traduction_sports = {
    'Archery': 'Tir à l\'arc',
    'Athletics': 'Athlétisme',
    'Badminton': 'Badminton',
    'Baseball': 'Baseball',
    'Basketball': 'Basketball',
    'Beach Volleyball': 'Beach Volley',
    'Boxing': 'Boxe',
    'Canoeing': 'Canoë-kayak',
    'Cycling': 'Cyclisme',
    'Diving': 'Plongeon',
    'Equestrianism': 'Équitation',
    'Fencing': 'Escrime',
    'Football': 'Football',
    'Golf': 'Golf',
    'Gymnastics': 'Gymnastique',
    'Handball': 'Handball',
    'Hockey': 'Hockey',
    'Ice Hockey': 'Hockey sur glace',
    'Judo': 'Judo',
    'Modern Pentathlon': 'Pentathlon moderne',
    'Rhythmic Gymnastics': 'Gymnastique rythmique',
    'Rugby': 'Rugby',
    'Sailing': 'Voile',
    'Shooting': 'Tir',
    'Swimming': 'Natation',
    'Synchronized Swimming': 'Natation synchronisée',
    'Table Tennis': 'Tennis de table',
    'Taekwondo': 'Taekwondo',
    'Tennis': 'Tennis',
    'Trampolining': 'Trampoline',
    'Triathlon': 'Triathlon',
    'Volleyball': 'Volleyball'
}
donnees['Sport'] = donnees['Sport'].map(traduction_sports)

# Encoder le genre
encodeur = LabelEncoder()
donnees['Sexe'] = encodeur.fit_transform(donnees['Sex'])

# Définir les sports pour 2024
sports_2024 = list(traduction_sports.values())

# Dictionnaire de modèles
modeles = {}

# Entraîner les modèles de régression logistique pour chaque sport
for sport in sports_2024:
    donnees_sport = donnees[donnees['Sport'] == sport]
    echantillons_positifs = donnees_sport[donnees_sport['Medal'] == 'G']
    echantillons_negatifs = donnees_sport[donnees_sport['Medal'] != 'G']
    if len(echantillons_positifs) < 2 or len(echantillons_negatifs) < 2:
        continue

    X = donnees_sport[['Sexe', 'Age', 'Height', 'Weight']].dropna()
    y = (donnees_sport['Medal'] == 'G').astype(int)
    modele = LogisticRegression()
    modele.fit(X, y)
    modeles[sport] = modele

# Fonction pour prédire les probabilités de médaille pour chaque sport basées sur les données de l'utilisateur
def calculer_probabilites_sport(donnees_utilisateur):
    probabilites_sports = {}
    for sport, modele in modeles.items():
        # Préparer les données de l'utilisateur pour la prédiction
        donnees_utilisateur_df = pd.DataFrame([donnees_utilisateur], columns=['Sexe', 'Age', 'Height', 'Weight'])
        # Imputer les valeurs manquantes
        imputer = SimpleImputer(strategy='mean')
        donnees_utilisateur_imputed = pd.DataFrame(imputer.fit_transform(donnees_utilisateur_df),
                                                    columns=donnees_utilisateur_df.columns)
        # Prédire la probabilité de gagner une médaille d'or pour le sport spécifique
        probabilite = modele.predict_proba(donnees_utilisateur_imputed)[:, 1]
        probabilites_sports[sport] = probabilite[0]
    # Trier les probabilités et sélectionner les 3 sports les plus probables
    probabilites_triees = sorted(probabilites_sports.items(), key=lambda x: x[1], reverse=True)
    top_sports = dict(probabilites_triees[:3])
    return top_sports

# Fonction pour afficher les sports les plus probables avec les probabilités les plus élevées
def obtenir_top_sports(probabilites):
    probabilites_triees = sorted(probabilites.items(), key=lambda x: x[1], reverse=True)
    top_3_sports = probabilites_triees[:3]
    top_sports = {sport: probabilite for sport, probabilite in top_3_sports}
    return top_sports

#INTERFACE --------------

# Initialiser Pygame
pygame.init()

largeur_ecran = 1400
hauteur_ecran = 800
ecran = pygame.display.set_mode((largeur_ecran, hauteur_ecran))
pygame.display.set_caption('Oracle Olympique')

# Définir les couleurs
BLANC = (255, 255, 255)
ROSE = (255, 20, 147)
GRIS_CLAIR = (220, 220, 220)
BLEU_CLAIR = (118, 189, 214)
NOIR = (0, 0, 0)
BLEU_MARINE = (6, 103, 162)

# Header 
photo_en_tete = pygame.image.load('banner.jpg')
photo_en_tete = pygame.transform.scale(photo_en_tete, (largeur_ecran, 200))

# definr les elt du header 
police_en_tete = pygame.font.Font(None, 80)
texte_en_tete = police_en_tete.render('Oracle Olympique', True, ROSE)
rect_en_tete = texte_en_tete.get_rect(center=(largeur_ecran // 2, 100))

#  les dimensions et le style de la zone saisie 
taille_boite_saisie = 300
couleur_boite_saisie = GRIS_CLAIR
couleur_bordure_boite_saisie = BLEU_CLAIR
couleur_texte_boite_saisie = NOIR
decalage_texte_boite_saisie = 10
couleur_boite_active = BLEU_MARINE

# def les boîtes de saisie pour Sexe, Âge, Taille et Poids
boites_saisie = []
labels_saisie = ['Sexe(F/M):', 'Âge:', 'Taille(cm):', 'Poids(kg):']
for i, label in enumerate(labels_saisie):
    boite = pygame.Rect(largeur_ecran - taille_boite_saisie - 50, 320 + i * 100, taille_boite_saisie, 60)
    boites_saisie.append((boite, label, '', False))

# Calc la position du bouton de soumission
rect_bouton_soumettre = pygame.Rect(largeur_ecran - taille_boite_saisie - 50, 320 + (len(boites_saisie) - 1) * 100 + 100, taille_boite_saisie, 60)
couleur_bouton_soumettre = BLEU_CLAIR
couleur_texte_bouton_soumettre = BLANC
police_bouton_soumettre = pygame.font.Font(None, 36)
texte_bouton_soumettre = police_bouton_soumettre.render('Valider', True, couleur_texte_bouton_soumettre)
rect_texte_bouton_soumettre = texte_bouton_soumettre.get_rect(center=rect_bouton_soumettre.center)

# Définir les boîtes de texte pour le titre et le sous-titre
boite_titre = pygame.Rect(50, 250, 400, 100)
boite_sous_titre = pygame.Rect(50, 450, 400, 50)

# Définir le texte du titre et du sous-titre
texte_titre = "ORACLE OLYMPIQUE"

#Def des subtitle 
texte_sous_titre1 = "Découvrez votre potentiel olympique avec Oracle Olympique !"
texte_sous_titre2 = "Notre programme révolutionnaire vous permet de plonger dans l'excitation"
texte_sous_titre3 = "des Jeux Olympiques 2024 en déterminant dans quelle discipline"
texte_sous_titre4 = "sportive vous pourriez briller et décrocher la médaille tant"
texte_sous_titre5 = "convoitée. Grâce à notre technologie avancée, comparez vos compétences"
texte_sous_titre6 = "et performances à celles des anciens athlètes pour savoir où"
texte_sous_titre7 = "vous vous situez. Ne laissez pas vos rêves de gloire"
texte_sous_titre8 = "olympique rester des rêves, laissez Oracle Olympique les réaliser"
texte_sous_titre9 = "pour vous. Découvrez votre destin olympique dès aujourd'hui !"
 

# Définir la police et les couleurs pour le texte du titre et du sous-titre
police_titre = pygame.font.Font(None, 60)
police_sous_titre = pygame.font.Font(None, 36)
couleur_texte = NOIR

# Fonction pour diviser le texte en lignes en fonction de la largeur maximale
def diviser_texte_en_lignes(texte, largeur_max):
    mots = texte.split()
    lignes = []
    ligne_actuelle = ''
    for mot in mots:
        if ligne_actuelle == '':
            ligne_actuelle = mot
        elif police_sous_titre.size(ligne_actuelle + ' ' + mot)[0] <= largeur_max:
            ligne_actuelle += ' ' + mot
        else:
            lignes.append(ligne_actuelle)
            ligne_actuelle = mot
    lignes.append(ligne_actuelle)
    return lignes


# Fonction pour convertir les données utilisateur en valeurs numériques
def convertir_donnees_numeriques(donnees_utilisateur):
    if len(donnees_utilisateur) > 0 and donnees_utilisateur[0] == 'M':
        donnees_utilisateur[0] = 0
    elif len(donnees_utilisateur) > 0:
        donnees_utilisateur[0] = 1
    
    for i in range(1, len(donnees_utilisateur)):
        if donnees_utilisateur[i] != '':
            donnees_utilisateur[i] = int(donnees_utilisateur[i])
        else:
            donnees_utilisateur[i] = np.nan


# Fonction pour afficher un message d'erreur
def afficher_message_erreur(message):
    police_erreur = pygame.font.Font(None, 36)
    texte_erreur = police_erreur.render(message, True, NOIR)
    rect_erreur = texte_erreur.get_rect(center=(largeur_ecran // 2, hauteur_ecran // 2))
    pygame.draw.rect(ecran, GRIS_CLAIR, rect_erreur.inflate(20, 20))
    pygame.draw.rect(ecran, BLEU_MARINE, rect_erreur, 2)
    ecran.blit(texte_erreur, rect_erreur)


def plot_sport_probabilities(probabilities, filename):
    plt.figure(figsize=(12, 6))
    plt.barh(list(probabilities.keys()), list(probabilities.values()), color='pink')
    plt.xlabel("Probabilité de médaille d'or")
    plt.title("Probabilités de médaille d'or pour les sports aux JO 2024")
    plt.gca().invert_yaxis()
    plt.savefig(filename)  # Save the plot as an image file

    # Open the saved image file and display it
    img = Image.open(filename)
    img.show()

# Initialiser Pygame et définir les variables d'écran
pygame.init()
largeur_ecran = 1400
hauteur_ecran = 800
ecran = pygame.display.set_mode((largeur_ecran, hauteur_ecran))
pygame.display.set_caption('Oracle Olympique')

    


running = True
message_erreur = None 
temps_erreur = 0  
duree_erreur = 2000 
donnees_utilisateur = []
while running:
    temps_actuel = pygame.time.get_ticks()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            for index, (boite, _, _, _) in enumerate(boites_saisie):
                if boite.collidepoint(event.pos):
                    for i, (autre_boite, label, texte, _) in enumerate(boites_saisie):
                        boites_saisie[i] = (autre_boite, label, texte, i == index)
                    break
            if rect_bouton_soumettre.collidepoint(event.pos):
                donnees_utilisateur.clear()
                for _, _, texte, _ in boites_saisie:
                    if texte == '':
                        message_erreur = "Vous n'avez pas entré toutes vos données !"
                        temps_erreur = temps_actuel + duree_erreur
                        break
                    donnees_utilisateur.append(texte)
                convertir_donnees_numeriques(donnees_utilisateur)
                probabilites_utilisateur = calculer_probabilites_sport(donnees_utilisateur)
                top_sports_utilisateur = obtenir_top_sports(probabilites_utilisateur)
                print(top_sports_utilisateur, probabilites_utilisateur)
                # Call the function to plot and display the sport probabilities
                plot_sport_probabilities(probabilites_utilisateur , 'sport_probabilities.png')

        elif event.type == pygame.KEYDOWN:
            index_boite_active = None
            for i, (_, _, _, actif) in enumerate(boites_saisie):
                if actif:
                    index_boite_active = i
                    break
            
            if index_boite_active is not None:
                boite_active = boites_saisie[index_boite_active]
                if event.key == pygame.K_BACKSPACE:
                    boites_saisie[index_boite_active] = (boite_active[0], boite_active[1], boite_active[2][:-1], True)
                elif event.key in (pygame.K_KP_ENTER, pygame.K_RETURN):
                    index_boite_suivante = (index_boite_active + 1) % len(boites_saisie)
                    for i, (boite, label, texte, _) in enumerate(boites_saisie):
                        boites_saisie[i] = (boite, label, texte, i == index_boite_suivante)
                else:
                    boites_saisie[index_boite_active] = (boite_active[0], boite_active[1], boite_active[2] + event.unicode, True)

    ecran.fill(BLANC)
    ecran.blit(photo_en_tete, (0, 0))

    for boite, label, texte, actif in boites_saisie:
        pygame.draw.rect(ecran, couleur_boite_saisie, boite)
        pygame.draw.rect(ecran, couleur_boite_active if actif else couleur_bordure_boite_saisie, boite, 2)
        police_label = pygame.font.Font(None, 36)
        surface_label = police_label.render(label, True, BLEU_MARINE)
        rect_label = surface_label.get_rect(midright=(boite.left - decalage_texte_boite_saisie, boite.centery))
        ecran.blit(surface_label, rect_label)
        police_saisie = pygame.font.Font(None, 36)
        surface_saisie = police_saisie.render(texte, True, couleur_texte_boite_saisie)
        rect_saisie = surface_saisie.get_rect(midleft=(boite.left + decalage_texte_boite_saisie, boite.centery))
        ecran.blit(surface_saisie, rect_saisie)

    pygame.draw.rect(ecran, couleur_bouton_soumettre, rect_bouton_soumettre)
    ecran.blit(texte_bouton_soumettre, rect_texte_bouton_soumettre)

    if message_erreur and temps_actuel < temps_erreur:
        afficher_message_erreur(message_erreur)

 
    texte_titre_surface = police_titre.render(texte_titre, True, couleur_texte)
    rect_titre = texte_titre_surface.get_rect(center=boite_titre.center)
    ecran.blit(texte_titre_surface, rect_titre)
    
    y_position = 450
    for sous_titre in [texte_sous_titre1, texte_sous_titre2, texte_sous_titre3, texte_sous_titre4, 
                   texte_sous_titre5, texte_sous_titre6, texte_sous_titre7, texte_sous_titre8, texte_sous_titre9]:
        texte_sous_titre_surface = police_sous_titre.render(sous_titre, True, couleur_texte)
        rect_sous_titre = texte_sous_titre_surface.get_rect(midleft=(50, y_position))
        ecran.blit(texte_sous_titre_surface, rect_sous_titre)
        y_position += texte_sous_titre_surface.get_height() + 10 
    pygame.display.flip()

pygame.quit()
