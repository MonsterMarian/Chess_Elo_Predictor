# Kód a texty pro Google Colab (Jupyter Notebook)

Pokud bys měl z jakéhokoliv důvodu problém s vygenerovaným `.ipynb` souborem, nebo by sis chtěl Colab naklikat úplně sám (buňku po buňce), tady máš připravený veškerý text a kód přesně tak, jak by měl jít za sebou.

Stačí si v Colabu zakládat buňky střídavě jako **Text** (Markdown) a **Code** (Kód) a vložit do nich tyto úseky.

---

### Buňka 1: Text
```markdown
# ♟️ Chess Elo Predictor - Tréninkový postup
Tento notebook dokumentuje proces tvorby modelu strojového učení pro predikci šachového ratingu na platformě Chess.com. Modely pak byly nasazeny do plnohodnotné webové aplikace. Tento dokument slouží jako ukázka toho, jak probíhal výzkum, sběr dat, předzpracování (čištění) a samotný trénink.
```

### Buňka 2: Text
```markdown
## 1. Instalace podružných knihoven
Zajistíme, že máme nainstalované potřebné nástroje (LightGBM, Pandas, Scikit-learn).
```

### Buňka 3: Kód
```python
!pip install pandas numpy lightgbm scikit-learn matplotlib
```

### Buňka 4: Text
```markdown
## 2. Načtení reálných dat
Data byla sesbírána vlastním napsaným asynchronním crawlerem (`downloader.py`), který z veřejného API Chess.com stáhl měsíční snapshoty pro stovky hráčů. Dataset obsahuje přes 1500 historií záznamů. Připojíme se k lokální SQLite databázi a vytáhneme data k trénování.
```

### Buňka 5: Kód
```python
import sqlite3
import pandas as pd
import numpy as np

# PŘED SPUŠTĚNÍM TÉTO BUŇKY: 
# Nahrajte soubor 'elo_history_v3.db' z vašeho PC do Google Colab (ikona složky vlevo -> Nahrát do úložiště relace).

# Připojíme se k opravdové databázi, kterou si náš crawler vytvořil po sběru z Chess.com API.
conn = sqlite3.connect('elo_history_v3.db')

# Stáhneme všechny nasbírané snapshoty z tabulky pomoci SQL dotazu
df_raw = pd.read_sql('SELECT * FROM player_snapshots', conn)

print(f"✅ Data úspěšně připojena. Načteno celkem {len(df_raw)} herních záznamů (snapshotů).")
# Ukaž prvních 10 řádek, aby učitel viděl reálná nasbíraná data různých uživatelů
df_raw.head(10)
```

### Buňka 6: Text
```markdown
## 3. Předzpracování dat (Data Preprocessing a Feature Engineering)
Surová data je potřeba přeložit do smysluplných atributů, učíme AI chápat časové trendy:
- `growth_3m`: Růst za poslední měsíce (Trend)
- `volatility`: Jak moc hráči Elo skáče (Standardní odchylka - stabilita hráče)
- `max_gap`: Největší herní výpadek (Měsíce bez hraní)  
Tyto atributy pomáhají modelu předpovídat lépe, než pouhé čisté stáhnutí původních Elo bodů.
```

### Buňka 7: Kód
```python
def build_features(player_df):
    df = player_df.sort_values('month_idx')
    elos = df['elo'].values
    months = df['month_idx'].values
    counts = df['games_count'].values
    
    if len(elos) < 2: return None
    
    window_elo = elos[-4:]
    growth = np.mean(np.diff(window_elo)) if len(window_elo) > 1 else 0
    volatility = np.std(window_elo)
    max_gap = np.max(np.diff(months)) if len(months) > 1 else 0
    
    # Z predikce nás zajímá, oč se mu Elo zvedlo (ne jeho absolutní hodnota)
    target_delta = elos[-1] - elos[-2]
    
    return {
        'current_elo': elos[-2],
        'month_idx': months[-2],
        'growth': growth,
        'volatility': volatility,
        'avg_activity': np.mean(counts),
        'max_gap': max_gap,
        'target_delta': target_delta
    }

features_list = []
for user, grp in df_raw.groupby('username'):
    if len(grp) >= 2:
        feat = build_features(grp)
        if feat: features_list.append(feat)

df_features = pd.DataFrame(features_list)
print("Vygenerované atributy (Features) připravené pro model:")
df_features.head()
```

### Buňka 8: Text
```markdown
## 4. Trénink modelu umělé inteligence (LightGBM)
Rozdělíme si vygenerovaná data na vstupy (`X`) a cíl `target_delta` (`y`). Poté na nich natrénujeme rozhodovací lesy skrze LightGBM, který je pro tento typ tabulkových dat s rozličnou křivkou růstu nejvhodnější.
```

### Buňka 9: Kód
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if len(df_features) > 0:
    X = df_features.drop('target_delta', axis=1)
    y = df_features['target_delta']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if len(X) > 5 else (X, X, y, y)
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✅ Model strojového učení úspěšně natrénován!")
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Chybovost modelu (MSE): {mse:.2f}")
else:
    print("Nedostatek dat pro trénink.")
```

### Buňka 10: Text
```markdown
## 5. Důležitost atributů (Feature Importance)
Můžeme se podívat do "hlavy" modelu, aby nám ukázal, na základě jakých metrik se primárně rozhoduje při predikci hráčovy budoucnosti.
```

### Buňka 11: Kód
```python
import matplotlib.pyplot as plt

if 'model' in locals():
    lgb.plot_importance(model, max_num_features=10, importance_type='split', title="Na co se AI dívá (Feature Importance)")
    plt.show()
```
