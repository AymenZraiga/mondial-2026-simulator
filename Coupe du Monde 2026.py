import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Données utilisées pour la simulation
# Les groupes sont fixés ici, puis chaque match est simulé à partir des Elo.
GROUPES = {
    "A": ["Mexique", "Corée du Sud", "Afrique du Sud", "Tchéquie"],
    "B": ["Canada", "Bosnie", "Qatar", "Suisse"],
    "C": ["Brésil", "Maroc", "Haïti", "Écosse"],
    "D": ["USA", "Paraguay", "Australie", "Turquie"],
    "E": ["Allemagne", "Curaçao", "Côte d'Ivoire", "Équateur"],
    "F": ["Pays-Bas", "Japon", "Suède", "Tunisie"],
    "G": ["Belgique", "Égypte", "Iran", "Nouvelle-Zélande"],
    "H": ["Espagne", "Cap-Vert", "Arabie Saoudite", "Uruguay"],
    "I": ["France", "Sénégal", "Irak", "Norvège"],
    "J": ["Argentine", "Algérie", "Autriche", "Jordanie"],
    "K": ["Portugal", "RD Congo", "Ouzbékistan", "Colombie"],
    "L": ["Angleterre", "Croatie", "Ghana", "Panama"],
}

# Scores Elo. Certaines valeurs peuvent être mises à jour si on veut refaire
# la simulation avec des données plus récentes.
ELO = {
    "Espagne": 2171, "Argentine": 2113, "France": 2063, "Angleterre": 2042,
    "Colombie": 1998, "Brésil": 1979, "Portugal": 1976, "Pays-Bas": 1959,
    "Croatie": 1933, "Équateur": 1933, "Norvège": 1922, "Allemagne": 1910,
    "Suisse": 1897, "Uruguay": 1890, "Maroc": 1888, "Turquie": 1880,
    "Japon": 1879, "Sénégal": 1869, "Mexique": 1865, "Belgique": 1849,
    "USA": 1828, "Canada": 1818, "Iran": 1817, "Suède": 1815,
    "Corée du Sud": 1812, "Autriche": 1805, "Algérie": 1802, "Tchéquie": 1800,
    "Côte d'Ivoire": 1798, "Écosse": 1790, "Égypte": 1788, "Australie": 1785,
    "Paraguay": 1760, "Ghana": 1758, "RD Congo": 1750, "Qatar": 1672,
    "Cap-Vert": 1668, "Ouzbékistan": 1660, "Arabie Saoudite": 1655, "Irak": 1642,
    "Afrique du Sud": 1640, "Jordanie": 1622, "Tunisie": 1700, "Bosnie": 1700,
    "Panama": 1690, "Nouvelle-Zélande": 1530, "Haïti": 1520, "Curaçao": 1480,
}

HOTES = {"USA", "Canada", "Mexique"}
BONUS_HOTE = 80

# Paramètres du modèle de buts.
# BASE donne le niveau moyen de buts, K règle l'effet de l'écart Elo.
BUTS_BASE = 1.30
K = 0.50


def buts_attendus(equipe_a, equipe_b):
    """Calcule les buts moyens attendus pour les deux équipes."""
    elo_a = ELO[equipe_a] + (BONUS_HOTE if equipe_a in HOTES else 0)
    elo_b = ELO[equipe_b] + (BONUS_HOTE if equipe_b in HOTES else 0)

    ecart = (elo_a - elo_b) / 400.0
    lam_a = BUTS_BASE * np.exp(K * ecart)
    lam_b = BUTS_BASE * np.exp(-K * ecart)
    return lam_a, lam_b


def simuler_match(equipe_a, equipe_b, elimination=False):
    """Simule un match. En phase finale, un nul est tranché aux tirs au but."""
    lam_a, lam_b = buts_attendus(equipe_a, equipe_b)
    buts_a = np.random.poisson(lam_a)
    buts_b = np.random.poisson(lam_b)

    if elimination and buts_a == buts_b:
        elo_a = ELO[equipe_a] + (BONUS_HOTE if equipe_a in HOTES else 0)
        elo_b = ELO[equipe_b] + (BONUS_HOTE if equipe_b in HOTES else 0)
        proba_a = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400.0))

        if np.random.random() < proba_a:
            buts_a += 1
        else:
            buts_b += 1

    return buts_a, buts_b


def simuler_groupe(equipes):
    """Simule les six matchs d'un groupe et renvoie le classement."""
    points = {equipe: 0 for equipe in equipes}
    diff_buts = {equipe: 0 for equipe in equipes}
    buts_marques = {equipe: 0 for equipe in equipes}

    for i in range(len(equipes)):
        for j in range(i + 1, len(equipes)):
            a, b = equipes[i], equipes[j]
            buts_a, buts_b = simuler_match(a, b)

            buts_marques[a] += buts_a
            buts_marques[b] += buts_b
            diff_buts[a] += buts_a - buts_b
            diff_buts[b] += buts_b - buts_a

            if buts_a > buts_b:
                points[a] += 3
            elif buts_b > buts_a:
                points[b] += 3
            else:
                points[a] += 1
                points[b] += 1

    classement = sorted(
        equipes,
        key=lambda e: (points[e], diff_buts[e], buts_marques[e], np.random.random()),
        reverse=True
    )

    troisieme = classement[2]
    stats_troisieme = (
        points[troisieme],
        diff_buts[troisieme],
        buts_marques[troisieme]
    )
    return classement, stats_troisieme


def construire_bracket(premiers, deuxiemes, troisiemes_qualifies):
    """Construit un tableau de phase finale simple à partir des qualifiés."""
    t = troisiemes_qualifies
    return [
        (premiers["A"], deuxiemes["B"]), (premiers["B"], deuxiemes["A"]),
        (premiers["C"], deuxiemes["D"]), (premiers["D"], deuxiemes["C"]),
        (premiers["E"], deuxiemes["F"]), (premiers["F"], deuxiemes["E"]),
        (premiers["G"], deuxiemes["H"]), (premiers["H"], deuxiemes["G"]),
        (premiers["I"], deuxiemes["J"]), (premiers["J"], deuxiemes["I"]),
        (premiers["K"], deuxiemes["L"]), (premiers["L"], deuxiemes["K"]),
        (t[0], t[7]), (t[1], t[6]), (t[2], t[5]), (t[3], t[4]),
    ]


def jouer_tour(paires):
    vainqueurs = []
    for a, b in paires:
        buts_a, buts_b = simuler_match(a, b, elimination=True)
        vainqueurs.append(a if buts_a > buts_b else b)
    return vainqueurs


def simuler_phase_finale(paires_r32, suivi):
    """Joue la phase finale et met à jour les compteurs de parcours."""
    for a, b in paires_r32:
        suivi[a]["R32"] += 1
        suivi[b]["R32"] += 1
    v32 = jouer_tour(paires_r32)

    for equipe in v32:
        suivi[equipe]["R16"] += 1
    paires_16 = [(v32[i], v32[i + 1]) for i in range(0, 16, 2)]
    v16 = jouer_tour(paires_16)

    for equipe in v16:
        suivi[equipe]["QF"] += 1
    paires_qf = [(v16[i], v16[i + 1]) for i in range(0, 8, 2)]
    vqf = jouer_tour(paires_qf)

    for equipe in vqf:
        suivi[equipe]["SF"] += 1
    paires_sf = [(vqf[i], vqf[i + 1]) for i in range(0, 4, 2)]
    vsf = jouer_tour(paires_sf)

    for equipe in vsf:
        suivi[equipe]["Finale"] += 1
    buts_a, buts_b = simuler_match(vsf[0], vsf[1], elimination=True)
    champion = vsf[0] if buts_a > buts_b else vsf[1]
    suivi[champion]["Champion"] += 1
    return champion


def simuler_tournoi(suivi):
    premiers = {}
    deuxiemes = {}
    troisiemes = []

    for groupe, equipes in GROUPES.items():
        classement, stats_3e = simuler_groupe(equipes)

        premiers[groupe] = classement[0]
        deuxiemes[groupe] = classement[1]

        troisiemes.append((classement[2], *stats_3e))

        suivi[classement[0]]["Qualifie"] += 1
        suivi[classement[1]]["Qualifie"] += 1

    troisiemes.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    meilleurs_3es = [ligne[0] for ligne in troisiemes[:8]]

    for equipe in meilleurs_3es:
        suivi[equipe]["Qualifie"] += 1

    bracket = construire_bracket(premiers, deuxiemes, meilleurs_3es)
    return simuler_phase_finale(bracket, suivi)


def monte_carlo(n_simulations=10000, graine=42):
    """Répète le tournoi plusieurs fois pour obtenir des probabilités."""
    np.random.seed(graine)
    stades = ["Qualifie", "R32", "R16", "QF", "SF", "Finale", "Champion"]
    suivi = {equipe: {stade: 0 for stade in stades} for equipe in ELO}

    for _ in range(n_simulations):
        simuler_tournoi(suivi)

    resultats = []
    for equipe in ELO:
        ligne = {"equipe": equipe, "elo": ELO[equipe]}
        for stade in stades:
            ligne[stade] = 100 * suivi[equipe][stade] / n_simulations
        resultats.append(ligne)

    return resultats


def afficher_favoris(lignes, n=14):
    favoris = sorted(lignes, key=lambda x: x["Champion"], reverse=True)[:n]

    print(f"\n{'Équipe':<16}{'Elo':>6}{'1/8':>9}{'Quarts':>9}"
          f"{'Demies':>9}{'Finale':>9}{'Titre':>9}")
    print("-" * 70)

    for ligne in favoris:
        print(f"{ligne['equipe']:<16}{ligne['elo']:>6}"
              f"{ligne['R16']:>8.1f}%{ligne['QF']:>8.1f}%"
              f"{ligne['SF']:>8.1f}%{ligne['Finale']:>8.1f}%"
              f"{ligne['Champion']:>8.1f}%")


def afficher_equipe(lignes, equipe):
    ligne = next(x for x in lignes if x["equipe"] == equipe)

    print(f"\n--- Focus {equipe} (Elo {ligne['elo']}) ---")
    print(f"Sort des poules : {ligne['Qualifie']:.1f}%")
    print(f"8es de finale   : {ligne['R16']:.1f}%")
    print(f"Quarts          : {ligne['QF']:.1f}%")
    print(f"Demies          : {ligne['SF']:.1f}%")
    print(f"Finale          : {ligne['Finale']:.1f}%")
    print(f"Vainqueur       : {ligne['Champion']:.1f}%")


NOTE_METHODO = (
    "Modèle : buts simulés avec une loi de Poisson selon l'écart Elo, "
    "puis 10 000 Coupes du Monde simulées.\n"
    "Remarque : les Elo peuvent être actualisés et le tableau final est simplifié."
)


def graphique_favoris(lignes, n=12, chemin="favoris_mondial_2026.png"):
    data = sorted(lignes, key=lambda x: x["Champion"], reverse=True)[:n]
    noms = [d["equipe"] for d in data][::-1]
    probas = [d["Champion"] for d in data][::-1]

    cmap = plt.get_cmap("YlGnBu")
    couleurs = [cmap(0.25 + 0.6 * (p / max(probas))) for p in probas]

    fig, ax = plt.subplots(figsize=(10, 7))
    barres = ax.barh(noms, probas, color=couleurs, edgecolor="white")

    for barre, proba in zip(barres, probas):
        ax.text(
            proba + max(probas) * 0.01,
            barre.get_y() + barre.get_height() / 2,
            f"{proba:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            color="#222"
        )

    ax.set_title(
        "Coupe du Monde 2026 : favoris selon la simulation",
        fontsize=18,
        fontweight="bold",
        pad=42,
        loc="left"
    )
    ax.text(
        0, 1.02,
        "Probabilité estimée de gagner le tournoi",
        transform=ax.transAxes,
        fontsize=12,
        color="#666"
    )

    ax.set_xlim(0, max(probas) * 1.15)
    ax.set_xlabel("Probabilité de titre (%)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.text(0.01, -0.02, NOTE_METHODO, fontsize=8, color="#888", ha="left")
    fig.tight_layout()
    fig.savefig(chemin, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chemin


def graphique_parcours(lignes, equipe="Maroc", chemin=None):
    if chemin is None:
        nom_fichier = equipe.lower().replace(" ", "_")
        chemin = f"parcours_{nom_fichier}.png"

    ligne = next(x for x in lignes if x["equipe"] == equipe)

    stades = ["Qualifie", "R16", "QF", "SF", "Finale", "Champion"]
    labels = ["Sortie des poules", "8es", "Quarts", "Demies", "Finale", "Titre"]
    valeurs = [ligne[stade] for stade in stades]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("Reds")
    couleurs = [cmap(0.35 + 0.5 * i / (len(stades) - 1)) for i in range(len(stades))]
    barres = ax.bar(labels, valeurs, color=couleurs, edgecolor="white", width=0.7)

    for barre, valeur in zip(barres, valeurs):
        ax.text(
            barre.get_x() + barre.get_width() / 2,
            valeur + max(valeurs) * 0.015,
            f"{valeur:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title(
        f"Parcours possible du {equipe}",
        fontsize=18,
        fontweight="bold",
        pad=42,
        loc="left"
    )
    ax.text(
        0, 1.02,
        f"Probabilité d'atteindre chaque tour selon la simulation (Elo {ligne['elo']})",
        transform=ax.transAxes,
        fontsize=12,
        color="#666"
    )

    ax.set_ylim(0, max(valeurs) * 1.15)
    ax.set_ylabel("Probabilité (%)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.text(0.01, -0.02, NOTE_METHODO, fontsize=8, color="#888", ha="left")
    fig.tight_layout()
    fig.savefig(chemin, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chemin


def main():
    n_simulations = 10000

    print("=" * 70)
    print(f"Simulation Coupe du Monde 2026 - {n_simulations} tournois")
    print("=" * 70)

    resultats = monte_carlo(n_simulations=n_simulations)

    print("\nFavoris de la simulation")
    afficher_favoris(resultats)

    afficher_equipe(resultats, "Maroc")

    image_favoris = graphique_favoris(resultats)
    image_maroc = graphique_parcours(resultats, "Maroc")

    print("\nGraphiques créés :")
    print(f"- {image_favoris}")
    print(f"- {image_maroc}")

    print("\nNote : les résultats dépendent des Elo utilisés, du bonus domicile")
    print("et du tableau de phase finale choisi pour cette simulation.")


if __name__ == "__main__":
    main()
