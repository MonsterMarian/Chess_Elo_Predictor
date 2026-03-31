# Budoucí Vylepšení — Chess Elo Predictor

Tento soubor obsahuje nápady na vylepšení, která nebyla implementována v aktuální verzi.

---

## 5. Backtesting a validace modelu

**Popis:**
Vzít historická data hráčů z databáze, "odříznout" posledních X měsíců, předpovědět je modelem a porovnat s realitou.
Výsledek zobrazit v UI jako: *"Model byl testován na N hráčích a měl průměrnou odchylku ±Y Elo."*

**Postup implementace:**
1. Z `snapshots` tabulky vytáhnout hráče s alespoň 6 měsíci dat.
2. Trénovat model pouze na prvních (n-3) měsících.
3. Předpovědět měsíce (n-2), (n-1), (n) a porovnat s reálnými hodnotami.
4. Vypočítat MAE (Mean Absolute Error) a RMSE.
5. Zobrazit jako "Přesnost modelu: ±X Elo (testováno na N hráčích)".

**Přínos:**
- Důvěryhodnost predikcí
- Možnost porovnávat různé verze modelu objektivně
- Pomáhá odhalit overfitting

**Odhadovaný dopad:** Nulový na přesnost, ale zásadní pro interpretaci výsledků.

---

## 10. Time-decay weighting při trénování

**Popis:**
Novější záznamy v databázi mají vyšší váhu při trénování. Styl hry hráče se mění — data stará 2 roky jsou pro predikci méně relevantní než data z posledních 3 měsíců.

**Postup:**
- LightGBM podporuje parametr `sample_weight` v `lgb.Dataset`
- Váha = `exp(-lambda * stáří_v_měsících)` kde lambda ~0.1

**Odhadovaný dopad:** +5–12% přesnosti pro krátkodobé predikce (1–3 měsíce).
