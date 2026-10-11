"""Formatage des nombres à la française pour le texte affiché dans le notebook.

Le formatage Python par défaut (``f"{x:.2f}"``) produit « 0.28 » et « 5,035 », alors que le
notebook est rédigé en français : « 0,28 » et « 5 035 ». Ces fonctions appliquent la
convention typographique française, sans dépendre des locales installées sur la machine du
jury (``locale.setlocale`` échoue si ``fr_FR`` n'est pas générée) :

- virgule décimale ;
- espace fine insécable (U+202F) comme séparateur de milliers et avant « % » et « € ».

Valeurs manquantes (``None``, ``NaN``) : rendues par « n.d. » plutôt que « nan ».
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pandas n'est importé qu'à l'usage (cf. configurer_pandas)
    import pandas as pd
    from pandas.io.formats.style import Styler

ESPACE_FINE = " "
NON_DISPONIBLE = "n.d."


def _manquant(valeur: float | None) -> bool:
    return valeur is None or (isinstance(valeur, float) and math.isnan(valeur))


def _franciser(texte: str) -> str:
    """Convertit un nombre formaté à l'anglaise (« 1,234.5 ») en « 1 234,5 »."""
    return texte.replace(",", ESPACE_FINE).replace(".", ",")


def nombre(valeur: float | None, decimales: int = 2, *, signe: bool = False) -> str:
    """Nombre décimal : ``nombre(0.2804)`` → « 0,28 », ``nombre(0.11, 4, signe=True)`` → « +0,1100 »."""
    if _manquant(valeur):
        return NON_DISPONIBLE
    assert valeur is not None
    spec = f"{'+' if signe else ''},.{decimales}f"
    return _franciser(format(valeur, spec))


def entier(valeur: float | None) -> str:
    """Entier avec séparateur de milliers : ``entier(5035)`` → « 5 035 »."""
    return nombre(valeur, 0)


def pourcentage(proportion: float | None, decimales: int = 1, *, signe: bool = False) -> str:
    """Proportion exprimée en pourcentage : ``pourcentage(0.28)`` → « 28,0 % »."""
    if _manquant(proportion):
        return NON_DISPONIBLE
    assert proportion is not None
    return f"{nombre(proportion * 100, decimales, signe=signe)}{ESPACE_FINE}%"


def euros(valeur: float | None, decimales: int = 0, *, signe: bool = False) -> str:
    """Montant en euros : ``euros(12345.6)`` → « 12 346 € »."""
    if _manquant(valeur):
        return NON_DISPONIBLE
    return f"{nombre(valeur, decimales, signe=signe)}{ESPACE_FINE}€"


def nombre_tableau(valeur: float | None, decimales_max: int = 4) -> str:
    """Nombre de cellule de tableau : au plus ``decimales_max`` décimales, zéros finaux retirés.

    ``nombre_tableau(0.28)`` → « 0,28 » ; ``nombre_tableau(5035.0)`` → « 5 035 » ;
    ``nombre_tableau(0.123456)`` → « 0,1235 ». Respecte ainsi un ``.round(2)`` fait en amont.
    """
    if _manquant(valeur):
        return NON_DISPONIBLE
    texte = nombre(valeur, decimales_max)
    if "," in texte:
        texte = texte.rstrip("0").rstrip(",")
    return "0" if texte in ("-0", "+0") else texte


# Pour un appel explicite de ``Styler.format(precision=...)`` SANS fonction de formatage : son
# paramètre ``decimal`` vaut « . » par défaut et écrase ``styler.format.decimal`` sur toutes
# les colonnes. Ne jamais le combiner avec une fonction de ce module : pandas retraiterait la
# chaîne déjà francisée (« 0,785 » → « 0 785 »). Utiliser ``styler_fr()``.
STYLER_FR: dict[str, str] = {"decimal": ",", "thousands": ESPACE_FINE}


def styler_fr(
    df: pd.DataFrame,
    formats: dict[str, Callable[[Any], str]] | None = None,
    precision: int | None = None,
) -> Styler:
    """``Styler`` au format français, avec des fonctions de formatage par colonne.

    Deux temps, car pandas ne sait pas combiner les deux proprement en un seul appel :

    1. ``STYLER_FR`` donne le format par défaut (virgule, espace fine) à **toutes** les
       colonnes ;
    2. les fonctions de ``formats`` (``nombre``, ``pourcentage``…) sont appliquées **sans**
       ``STYLER_FR`` et limitées à leurs colonnes (``subset``) : leur sortie, déjà en
       français, n'est pas retraitée et les autres colonnes gardent le format de l'étape 1.

    ``styler_fr(df, {"taux": pourcentage, "gain": euros})``
    """
    styler = df.style.format(precision=precision, **STYLER_FR)
    if formats:
        styler = styler.format(formats, subset=list(formats))
    return styler


def configurer_pandas() -> None:
    """Affiche les tableaux pandas au format français (virgule décimale, espace fine).

    - ``DataFrame`` affiché tel quel : ``display.float_format`` (flottants uniquement,
      pandas n'offre pas d'option équivalente pour les entiers) ;
    - ``Styler`` sans formateur explicite : ``styler.format.decimal`` et
      ``styler.format.thousands`` (flottants et entiers).
    """
    import pandas as pd

    pd.set_option("display.float_format", nombre_tableau)
    pd.set_option("styler.format.decimal", STYLER_FR["decimal"])
    pd.set_option("styler.format.thousands", STYLER_FR["thousands"])


def scientifique(valeur: float | None, decimales: int = 2) -> str:
    """Notation scientifique (p-values) : ``scientifique(1.234e-5)`` → « 1,23e-05 ».

    Une p-value nulle est un sous-dépassement flottant (scipy renvoie 0 sous ~1e-308), pas
    une certitude : elle s'affiche « < 1e-300 » plutôt que « 0,00e+00 ».
    """
    if _manquant(valeur):
        return NON_DISPONIBLE
    assert valeur is not None
    if valeur == 0:
        return "< 1e-300"
    return format(valeur, f".{decimales}e").replace(".", ",")
