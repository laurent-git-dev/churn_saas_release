"""Tests du formatage des nombres à la française (churn_saas.format_fr)."""

import numpy as np
import pytest

from churn_saas.format_fr import (
    ESPACE_FINE,
    NON_DISPONIBLE,
    configurer_pandas,
    entier,
    euros,
    nombre,
    nombre_tableau,
    pourcentage,
    scientifique,
    styler_fr,
)

E = ESPACE_FINE


class TestNombre:
    def test_virgule_decimale(self):
        assert nombre(0.2804) == "0,28"

    def test_separateur_milliers(self):
        assert nombre(1234567.891, 1) == f"1{E}234{E}567,9"

    def test_signe_explicite(self):
        assert nombre(0.1147, 4, signe=True) == "+0,1147"
        assert nombre(-0.5, 1, signe=True) == "-0,5"

    def test_negatif(self):
        assert nombre(-1234.5, 1) == f"-1{E}234,5"

    @pytest.mark.parametrize("manquant", [None, float("nan"), np.nan])
    def test_valeur_manquante(self, manquant):
        assert nombre(manquant) == NON_DISPONIBLE


class TestEntier:
    def test_milliers(self):
        assert entier(5035) == f"5{E}035"

    def test_petit_entier(self):
        assert entier(35) == "35"

    def test_numpy(self):
        assert entier(np.int64(5000)) == f"5{E}000"


class TestPourcentage:
    def test_proportion(self):
        assert pourcentage(0.28) == f"28,0{E}%"

    def test_decimales(self):
        assert pourcentage(0.12345, 2) == f"12,35{E}%"

    def test_signe(self):
        assert pourcentage(0.021, signe=True) == f"+2,1{E}%"

    def test_manquant(self):
        assert pourcentage(None) == NON_DISPONIBLE


class TestEuros:
    def test_arrondi_entier(self):
        assert euros(12345.6) == f"12{E}346{E}€"

    def test_decimales(self):
        assert euros(9.5, 2) == f"9,50{E}€"

    def test_signe(self):
        assert euros(1500, signe=True) == f"+1{E}500{E}€"


class TestNombreTableau:
    def test_zeros_finaux_retires(self):
        assert nombre_tableau(0.28) == "0,28"

    def test_quatre_decimales_max(self):
        assert nombre_tableau(0.123456) == "0,1235"

    def test_valeur_entiere(self):
        assert nombre_tableau(5035.0) == f"5{E}035"

    def test_zero_negatif(self):
        assert nombre_tableau(-0.00001) == "0"

    def test_manquant(self):
        assert nombre_tableau(float("nan")) == NON_DISPONIBLE


class TestConfigurerPandas:
    def test_dataframe_et_styler(self):
        import pandas as pd

        df = pd.DataFrame({"x": [0.25, 1234.5], "n": [5035, 12]})
        configurer_pandas()
        try:
            assert "0,25" in df.to_string()
            assert f"1{E}234,5" in df.to_string()
            assert f"5{E}035" in df.style.to_html()
        finally:
            pd.reset_option("display.float_format")
            pd.reset_option("styler.format.decimal")
            pd.reset_option("styler.format.thousands")


class TestGraduationsFigures:
    def test_libelles(self):
        from churn_saas.viz import _franciser_graduation

        assert _franciser_graduation("0.25") == "0,25"
        assert _franciser_graduation("12000") == f"12{E}000"
        assert _franciser_graduation("1500.5") == f"1{E}500,5"
        assert _franciser_graduation("40%") == f"40{E}%"
        assert _franciser_graduation("−0.5") == "−0,5"

    def test_axes_numeriques_seulement(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from churn_saas.viz import _FormateurFrancais, _franciser_axes

        fig, ax = plt.subplots()
        ax.bar(["a", "b"], [0.25, 0.75])  # x catégoriel, y numérique
        _franciser_axes(fig)
        _franciser_axes(fig)  # idempotent : pas de double enveloppe
        fig.canvas.draw()
        assert not isinstance(ax.xaxis.get_major_formatter(), _FormateurFrancais)
        formateur_y = ax.yaxis.get_major_formatter()
        assert isinstance(formateur_y, _FormateurFrancais)
        assert not isinstance(formateur_y.base, _FormateurFrancais)
        libelles = [t.get_text() for t in ax.get_yticklabels()]
        assert "0,2" in libelles and not any("." in lib for lib in libelles)
        plt.close(fig)


class TestScientifique:
    def test_p_value(self):
        assert scientifique(1.234e-5) == "1,23e-05"

    def test_manquant(self):
        assert scientifique(float("nan")) == NON_DISPONIBLE

    def test_zero_sous_depassement(self):
        assert scientifique(0.0) == "< 1e-300"


class TestStylerFr:
    @staticmethod
    def _cellules(styler) -> list[str]:
        import re

        return re.findall(r"<td[^>]*>([^<]*)</td>", styler.to_html())

    def test_fonctions_francaises_non_retraitees(self):
        # Régression : Styler.format(fonction, decimal=",", thousands=…) transformait
        # « 0,785 » en « 0 785 » et « 28,0 % » en « 28 0 % »
        import pandas as pd

        df = pd.DataFrame({"auc": [0.785], "taux": [0.28], "n": [5035]})
        cellules = self._cellules(
            styler_fr(df, {"auc": lambda v: nombre(v, 3), "taux": pourcentage, "n": entier})
        )
        assert cellules == ["0,785", f"28,0{E}%", f"5{E}035"]

    def test_colonnes_sans_fonction_au_format_francais(self):
        import pandas as pd

        df = pd.DataFrame({"auc": [0.785], "gain": [1234.5678]})
        cellules = self._cellules(styler_fr(df, {"auc": lambda v: nombre(v, 3)}, precision=2))
        assert cellules == ["0,785", f"1{E}234,57"]
