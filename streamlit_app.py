# Fichier: solveur_ro.py
# Module de Recherche Opérationnelle (RO) utilisant la librairie PuLP pour l'optimisation.

from pulp import *
import numpy as np

# --- HYPOTHÈSES ET COÛTS ---
# (Ces valeurs seraient lues depuis une base de données en production réelle)
COUT_STOCK_UNITAIRE = 5.0    # Coût de stockage d'une pièce pendant une période (DIN/jour)
COUT_RUPTURE_UNITAIRE = 50.0 # Coût estimé d'une rupture de stock (perte de production, urgence)
DEMANDE_MOYENNE_JOUR = 0.5   # En moyenne, 0.5 pièce est nécessaire par jour
ECART_TYPE_DEMANDE = 0.1    

def optimiser_stock_securite(
        stock_actuel: int, 
        delai_fournisseur: int, 
        probabilite_rupture_ia: float, 
        probabilite_panne_ia: float
) -> dict:
    """
    Calcule la quantité Q à commander et le niveau de stock de sécurité optimal (S).
    
    Arguments:
        stock_actuel: Niveau de stock actuel de la pièce de rechange.
        delai_fournisseur: Délai de livraison en jours.
        probabilite_rupture_ia: Probabilité de rupture prédite par l'IA (entre 0 et 1).
        probabilite_panne_ia: Probabilité de panne prédite par l'IA (entre 0 et 1).
        
    Retourne:
        Un dictionnaire avec la décision et les métriques optimisées.
    """
    
    # 1. AJUSTEMENT DE LA DEMANDE PAR LE RISQUE DE PANNE (Lien IA <-> RO)
    # Si le risque de panne est élevé, la demande future augmente, même si la pièce est rarement utilisée.
    # On multiplie la demande moyenne par un facteur basé sur le risque de panne (jusqu'à 1.5x)
    facteur_risque = 1.0 + (probabilite_panne_ia * 0.5) 
    demande_ajustee = DEMANDE_MOYENNE_JOUR * facteur_risque

    # 2. CALCUL DU STOCK CIBLE (basé sur la demande ajustée pendant le délai)
    stock_cible = demande_ajustee * delai_fournisseur
    
    # 3. UTILISATION DE L'OPTIMISATION LINÉAIRE POUR TROUVER LE STOCK DE SÉCURITÉ OPTIMAL (S)
    
    # Création du problème d'optimisation
    prob = LpProblem("Optimisation_Stock", LpMinimize)
    
    # Variable de décision : S, le stock de sécurité optimal
    S = LpVariable("Stock_Securite", lowBound=0, cat='Continuous')

    # Objectif : Minimiser (Coût de Stockage + Coût de Rupture PONDÉRÉ par la Probabilité IA)
    # L'IA fait monter le Coût de Rupture, forçant la RO à augmenter S.
    
    # Le coût de stockage est proportionnel au stock de sécurité S
    cout_stockage = S * COUT_STOCK_UNITAIRE
    
    # Le coût de rupture est proportionnel au risque IA (probabilité)
    # Note : Le terme exact est complexe (fonction de perte), nous utilisons ici une approximation simplifiée
    # pour illustrer le lien RO-IA dans PuLP.
    cout_rupture_pondere = COUT_RUPTURE_UNITAIRE * probabilite_rupture_ia * (1 / (S + 1e-6)) # Plus S est petit, plus le coût pondéré est haut

    # Fonction Objectif
    prob += cout_stockage + cout_rupture_pondere, "Minimisation_Couts_Totaux"
    
    # Contrainte de service minimum (simplifiée : S doit couvrir au moins 90% de la demande moyenne du délai)
    prob += S >= stock_cible * 0.9, "Contrainte_Service_Minimum"

    # Résolution du problème
    prob.solve(PULP_CBC_CMD(msg=0)) # msg=0 pour ne pas afficher le solveur
    
    stock_securite_optimal = value(S)
    
    # 4. DÉCISION FINALE (Quantité à commander)
    Q_a_commander = int(max(0, np.ceil(stock_securite_optimal) - stock_actuel))
    
    # 5. GÉNÉRATION DE LA RECOMMANDATION
    reco_finale = ""
    if Q_a_commander > 0:
        reco_finale = f"Commander **{Q_a_commander}** unités pour atteindre le stock de sécurité optimal de {stock_securite_optimal:.1f} pièces."
        if probabilite_panne_ia > 0.7 or probabilite_rupture_ia > 0.7:
             reco_finale += " (⚠️ Priorité : URGENTE, en raison du haut risque IA)"
    else:
        reco_finale = f"Aucune commande nécessaire. Le stock de sécurité optimal ({stock_securite_optimal:.1f} pièces) est atteint."

    return {
        "stock_securite_optimal": stock_securite_optimal,
        "quantite_a_commander": Q_a_commander,
        "demande_ajustee_jour": demande_ajustee,
        "recommandation_ro": reco_finale,
        "statut_solveur": LpStatus[prob.status]
    }

# Exemple de test (non exécuté dans le streamlit, juste pour le fichier)
# if __name__ == '__main__':
#     # Scénario critique : haute probabilité de rupture/panne, stock faible, long délai
#     resultat = optimiser_stock_securite(
#         stock_actuel=2,
#         delai_fournisseur=15,
#         probabilite_rupture_ia=0.85, 
#         probabilite_panne_ia=0.75 
#     )
#     print(resultat)
