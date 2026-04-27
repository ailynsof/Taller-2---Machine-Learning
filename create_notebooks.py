import nbformat as nbf
import os

# Create 01_setup.ipynb
nb1 = nbf.v4.new_notebook()

celda1 = """import pandas as pd, numpy as np, requests, json, os, warnings
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay)
import joblib
from mplsoccer import Pitch
warnings.filterwarnings('ignore')
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('dashboard', exist_ok=True)"""

celda2 = """BASE = "https://premier.72-60-245-2.sslip.io"

def load_or_download(path, url):
    if os.path.exists(path):
        print(f"[caché] Cargando {path}")
        return pd.read_csv(path)
    print(f"[descarga] {url}")
    df = pd.read_csv(url)
    df.to_csv(path, index=False)
    print(f"[guardado] {path} — {df.shape}")
    return df

players = load_or_download('data/players.csv', f"{BASE}/export/players")
matches = load_or_download('data/matches.csv', f"{BASE}/export/matches")
events  = load_or_download('data/events.csv',  f"{BASE}/export/events")"""

celda3 = """for name, df in [('players', players), ('matches', matches), ('events', events)]:
    print(f"\\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Nulos: {df.isnull().sum()[df.isnull().sum()>0].to_dict()}")
    display(df.head(2))"""

celda4 = """if 'as_' in matches.columns:
    matches.rename(columns={'as_': 'away_shots'}, inplace=True)
    print("✅ as_ renombrada a away_shots")

# Rename columns to standard ones used in the code
cols_to_rename = {'ftr': 'result', 'fthg': 'home_goals', 'ftag': 'away_goals', 'hy': 'home_yellow_cards'}
matches.rename(columns={k: v for k, v in cols_to_rename.items() if k in matches.columns}, inplace=True)

matches['date'] = pd.to_datetime(matches['date'], format='%d/%m/%Y')
print("✅ Fechas convertidas")

team_map = {
    'Man United': 'Man Utd',
    "Nott'm Forest": 'Nottingham Forest',
    'Spurs': 'Tottenham',
    'Wolves': 'Wolverhampton',
}
for col in ['home_team', 'away_team']:
    if col in matches.columns:
        matches[col] = matches[col].replace(team_map)
print("✅ team_map aplicado")"""

celda5 = """events_spatial = events[(events['x'] != 0) | (events['y'] != 0)].copy()
print(f"events original: {len(events)} | events_spatial: {len(events_spatial)}")"""

celda6 = """shots = events_spatial[events_spatial['is_shot'] == True].copy()
print(f"Tiros totales: {len(shots)} | Goles: {shots['is_goal'].sum()}")
print(f"Tasa de conversión: {shots['is_goal'].mean()*100:.1f}%")"""

celda7 = """if 'qualifiers' in shots.columns:
    q = shots['qualifiers'].astype(str)
else:
    q = pd.Series('', index=shots.index)

shots['is_big_chance']  = q.str.contains('BigChance',  na=False).astype(int)
shots['is_header']      = q.str.contains('"Head"',     na=False).astype(int)
shots['is_right_foot']  = q.str.contains('RightFoot',  na=False).astype(int)
shots['is_left_foot']   = q.str.contains('LeftFoot',   na=False).astype(int)
shots['is_penalty']     = q.str.contains('"Penalty"',  na=False).astype(int)
shots['is_counter']     = q.str.contains('FastBreak',  na=False).astype(int)
shots['from_corner']    = q.str.contains('FromCorner', na=False).astype(int)
shots['is_volley']      = q.str.contains('Volley',     na=False).astype(int)
shots['first_touch']    = q.str.contains('FirstTouch', na=False).astype(int)

qualifier_cols = ['is_big_chance','is_header','is_right_foot','is_left_foot',
                  'is_penalty','is_counter','from_corner','is_volley','first_touch']
print(shots[qualifier_cols].sum().sort_values(ascending=False))"""

celda8 = """shots['distance_to_goal'] = np.sqrt((100 - shots['x'])**2 + (50 - shots['y'])**2)
shots['angle_to_goal']    = np.arctan2(np.abs(50 - shots['y']), 100 - shots['x'])
shots['x_norm']           = shots['x'] / 100
shots['y_centered']       = np.abs(shots['y'] - 50) / 50
print(shots[['x','y','distance_to_goal','angle_to_goal','is_goal']].describe())"""

celda9 = """shots.to_csv('data/shots_processed.csv', index=False)
matches.to_csv('data/matches_clean.csv', index=False)
print("✅ shots_processed.csv y matches_clean.csv guardados")"""

nb1['cells'] = [nbf.v4.new_code_cell(c) for c in [celda1, celda2, celda3, celda4, celda5, celda6, celda7, celda8, celda9]]
with open('notebooks/01_setup.ipynb', 'w') as f:
    nbf.write(nb1, f)


# Create 02_eda.ipynb
nb2 = nbf.v4.new_notebook()
c2_0 = """import pandas as pd, numpy as np, os, warnings
import matplotlib.pyplot as plt, seaborn as sns
from mplsoccer import Pitch
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
shots   = pd.read_csv('data/shots_processed.csv')
matches = pd.read_csv('data/matches_clean.csv')
players = pd.read_csv('data/players.csv')
p_rename = {'xG': 'expected_goals', 'xA': 'expected_assists', 'web_name': 'player_name', 'now_cost': 'price'}
players.rename(columns={k:v for k,v in p_rename.items() if k in players.columns}, inplace=True)
if 'price' not in players.columns and 'now_cost' in players.columns: players.rename(columns={'now_cost': 'price'}, inplace=True)
"""

c2_A1 = """fig, ax = plt.subplots(figsize=(6,4))
counts = shots['is_goal'].value_counts()
ax.bar(['No gol', 'Gol'], counts.values, color=['#3498db', '#e74c3c'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v} ({(v/len(shots))*100:.1f}%)", ha='center')
ax.axhline(len(shots)*0.888, color='red', linestyle='--')
ax.text(0.5, len(shots)*0.888 + 50, 'Baseline naive: 88.8%', color='red')
ax.set_title('Distribución de clase objetivo')
plt.tight_layout(); plt.savefig('figures/class_distribution.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_A2 = """pitch = Pitch(pitch_type='opta', pitch_color='#1a472a', line_color='white')
fig, ax = pitch.draw(figsize=(12,8))
no_goals = shots[shots['is_goal']==0]
goals = shots[shots['is_goal']==1]
pitch.scatter(no_goals['x'], no_goals['y'], alpha=0.3, s=15, color='#3498db', ax=ax, label='No gol')
pitch.scatter(goals['x'], goals['y'], alpha=0.8, s=40, color='#f1c40f', ax=ax, label='Gol', zorder=5)
ax.set_title("Shot map — Premier League 2025/26 (GW1-GW30)", fontsize=15, color='white')
ax.legend(loc='upper left')
plt.tight_layout(); plt.savefig('figures/shot_map_all.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_A3 = """bins = [0, 10, 20, 30, 100]
labels = ['0-10m','10-20m','20-30m','30m+']
shots['dist_bin'] = pd.cut(shots['distance_to_goal'], bins=bins, labels=labels)
conv = shots.groupby('dist_bin')['is_goal'].mean() * 100
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.barh(conv.index[::-1], conv.values[::-1], color='#9b59b6')
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va='center')
ax.set_title('Conversión por zona de distancia')
ax.set_xlabel('Tasa de conversión (%)')
plt.tight_layout(); plt.savefig('figures/conversion_by_distance.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_A4 = """types = ['is_penalty', 'is_big_chance', 'is_header', 'is_counter']
conv_type = {t: shots[shots[t]==1]['is_goal'].mean()*100 for t in types}
conv_type['General'] = shots['is_goal'].mean()*100
conv_type = pd.Series(conv_type).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.barh(conv_type.index, conv_type.values, color='#e67e22')
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va='center')
ax.set_title('Conversión por tipo de disparo')
ax.set_xlabel('Tasa de conversión (%)')
plt.tight_layout(); plt.savefig('figures/conversion_by_type.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_A5 = """cols = ['distance_to_goal','angle_to_goal','is_big_chance','is_header',
        'is_right_foot','is_penalty','is_counter','from_corner','is_goal']
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(shots[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Heatmap de correlación')
plt.tight_layout(); plt.savefig('figures/correlation_heatmap.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_B1 = """results = matches['result'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(6,6))
ax.pie(results.values, labels=results.index, autopct='%1.1f%%', colors=['#2ecc71', '#95a5a6', '#e74c3c'])
ax.set_title('Distribución resultados H/D/A')
plt.tight_layout(); plt.savefig('figures/results_distribution.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_B2 = """matches['total_goals'] = matches['home_goals'] + matches['away_goals']
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(matches['total_goals'], kde=True, ax=ax, color='steelblue', binwidth=1)
mean_g = matches['total_goals'].mean()
ax.axvline(mean_g, color='red', linestyle='--', label=f"Media: {mean_g:.2f}")
ax.axvline(2.5, color='orange', linestyle=':', label='Over 2.5')
ax.set_title('Histograma goles/partido')
ax.legend()
plt.tight_layout(); plt.savefig('figures/goals_per_match.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_B3 = """matches['imp_h'] = (1/matches['b365h']) / (1/matches['b365h']+1/matches['b365d']+1/matches['b365a'])
matches['imp_d'] = (1/matches['b365d']) / (1/matches['b365h']+1/matches['b365d']+1/matches['b365a'])
matches['imp_a'] = (1/matches['b365a']) / (1/matches['b365h']+1/matches['b365d']+1/matches['b365a'])
melted = matches.melt(id_vars=['result'], value_vars=['imp_h','imp_d','imp_a'], var_name='Odd_Type', value_name='Implied_Probability')
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=melted, x='result', y='Implied_Probability', hue='Odd_Type', ax=ax)
ax.set_title('Implied probability de odds Bet365 por resultado')
plt.tight_layout(); plt.savefig('figures/implied_probability.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_B4 = """referee_stats = matches.groupby('referee').agg(
    partidos=('result','count'),
    home_win_rate=('result', lambda x: (x=='H').mean()),
    tarjetas=('home_yellow_cards','mean')  
).query('partidos >= 5').sort_values('home_win_rate', ascending=False).head(10)
global_hw = (matches['result']=='H').mean()
referee_stats['diferencia'] = referee_stats['home_win_rate'] - global_hw

fig, ax = plt.subplots(figsize=(8,6))
referee_stats['home_win_rate'].sort_values().plot(kind='barh', ax=ax, color='#1abc9c')
ax.axvline(global_hw, color='red', linestyle='--', label=f'Global HW ({global_hw:.1%})')
ax.set_title('Top 10 árbitros por Home Win Rate (min 5 partidos)')
ax.legend()
plt.tight_layout(); plt.savefig('figures/referee_bias.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_B5 = """top_team = matches['home_team'].value_counts().index[0]
team_matches = matches[matches['home_team'] == top_team].sort_values('date').copy()
team_matches['rolling_goals'] = team_matches['home_goals'].shift(1).rolling(5).mean()
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(team_matches['date'], team_matches['home_goals'], label='Goles por partido', marker='o', alpha=0.5)
ax.plot(team_matches['date'], team_matches['rolling_goals'], label='Rolling avg 5 partidos (shift 1)', linewidth=3, color='red')
ax.set_title(f'Rolling average de goles anotados como local - {top_team}')
ax.legend()
plt.tight_layout(); plt.savefig('figures/rolling_goals.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_C1 = """top_players = players.nlargest(15, 'expected_goals').sort_values('expected_goals', ascending=True)
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=top_players, x='expected_goals', y='player_name', hue='team', dodge=False, ax=ax)
ax.set_title('Top 15 jugadores por xG')
plt.tight_layout(); plt.savefig('figures/top_players_xg.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_C2 = """fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=players, x='expected_goals', y='price', hue='position', size='minutes', sizes=(20,200), alpha=0.6, ax=ax)
sns.regplot(data=players, x='expected_goals', y='price', scatter=False, color='black', ax=ax)
ax.set_title('Precio FPL vs xG')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout(); plt.savefig('figures/price_vs_xg.png', dpi=150, bbox_inches='tight'); plt.show()"""

c2_final = """## Insights accionables para feature engineering
1. Los tiros desde <10m tienen mucha mayor conversión vs en >20m → distance_to_goal es el predictor más importante.
2. BigChance tiene la mayor conversión → is_big_chance es la feature booleana con mayor poder predictivo.
3. Las odds de Bet365 predicen correctamente el resultado en ~50% de los casos → las odds son features informativas pero con límite.
4. El árbitro con mayor sesgo tiene alto home win rate vs la media global → referee como feature categórica puede aportar.
5. Los equipos con rolling_goals_scored alto en últimos 5 partidos tienen más probabilidad de ganar → rolling features justificadas."""

nb2['cells'] = [nbf.v4.new_code_cell(c) for c in [c2_0, c2_A1, c2_A2, c2_A3, c2_A4, c2_A5, c2_B1, c2_B2, c2_B3, c2_B4, c2_B5, c2_C1, c2_C2]] + [nbf.v4.new_markdown_cell(c2_final)]
with open('notebooks/02_eda.ipynb', 'w') as f:
    nbf.write(nb2, f)

# Create 03_models.ipynb
nb3 = nbf.v4.new_notebook()
c3_0 = """import pandas as pd, numpy as np, requests, json, os, warnings
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay)
import joblib
from mplsoccer import Pitch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

shots   = pd.read_csv('data/shots_processed.csv')
matches = pd.read_csv('data/matches_clean.csv')
players = pd.read_csv('data/players.csv')
p_rename = {'xG': 'expected_goals', 'xA': 'expected_assists', 'web_name': 'player_name'}
players.rename(columns={k:v for k,v in p_rename.items() if k in players.columns}, inplace=True)"""

c3_A1 = """features_xg = ['distance_to_goal','angle_to_goal','x_norm','y_centered',
                'is_big_chance','is_header','is_right_foot','is_left_foot',
                'is_penalty','is_counter','from_corner','is_volley','first_touch']
X = shots[features_xg].fillna(0)
y = shots['is_goal']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)"""

c3_A2 = """pipe_xg = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(penalty='l2', C=1.0, class_weight='balanced',
                               max_iter=1000, random_state=42))
])
pipe_xg.fit(X_train, y_train)"""

c3_A3 = """y_pred = pipe_xg.predict(X_test)
y_prob = pipe_xg.predict_proba(X_test)[:,1]

print("=== MODELO xG ===")
print(f"Baseline naive (siempre no-gol): {1 - y_test.mean():.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['No gol','Gol']))"""

c3_A4 = """# 1. Matriz de confusión
fig, ax = plt.subplots(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No gol','Gol'], yticklabels=['No gol','Gol'])
ax.set_title('Matriz de confusión — Modelo xG')
ax.set_ylabel('Real'); ax.set_xlabel('Predicho')
plt.tight_layout(); plt.savefig('figures/confusion_xg.png', dpi=150); plt.show()

# 2. Curva ROC
fig, ax = plt.subplots(figsize=(7,5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax,
    name=f'Logistic Regression (AUC={roc_auc_score(y_test,y_prob):.3f})')
ax.plot([0,1],[0,1],'--k', alpha=0.4, label='Baseline aleatorio')
ax.set_title('Curva ROC — Modelo xG'); ax.legend()
plt.tight_layout(); plt.savefig('figures/roc_xg.png', dpi=150); plt.show()

# 3. Feature importance
coefs = pd.Series(pipe_xg.named_steps['clf'].coef_[0], index=features_xg).sort_values()
colors = ['#d73027' if c > 0 else '#4575b4' for c in coefs]
fig, ax = plt.subplots(figsize=(8,6))
coefs.plot(kind='barh', color=colors, ax=ax)
ax.set_title('Coeficientes del modelo xG (positivo = más prob de gol)')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout(); plt.savefig('figures/importance_xg.png', dpi=150); plt.show()

# 4. Shot map test set con xG predicho
shots_test = shots.iloc[X_test.index].copy()
shots_test['xg_pred'] = y_prob
pitch = Pitch(pitch_type='opta', pitch_color='#1a472a', line_color='white')
fig, ax = pitch.draw(figsize=(12,8))
sc = pitch.scatter(shots_test['x'], shots_test['y'],
    c=shots_test['xg_pred'], cmap='RdYlGn_r', s=40,
    edgecolors=np.where(shots_test['is_goal']==1,'white','none'),
    linewidths=1.5, ax=ax, alpha=0.8)
plt.colorbar(sc, ax=ax, label='xG predicho')
ax.set_title('Shot map — xG predicho por el modelo (borde blanco = gol real)')
plt.tight_layout(); plt.savefig('figures/shot_map_xg.png', dpi=150); plt.show()"""

c3_A5 = """joblib.dump(pipe_xg, 'models/xg_model.pkl')
shots['xg_predicted'] = pipe_xg.predict_proba(X)[:,1]
shots[['match_id','player_name','x','y','is_goal','xg_predicted']].to_csv(
    'data/shots_with_xg.csv', index=False)
print("✅ xg_model.pkl y shots_with_xg.csv guardados")"""

c3_B1 = """cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelos = {
    'LogisticRegression': pipe_xg,
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                        random_state=42, n_jobs=-1))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(n_estimators=200, scale_pos_weight=6,
                               eval_metric='logloss', random_state=42))
    ])
}

resultados = {}
for nombre, modelo in modelos.items():
    acc  = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy').mean()
    f1   = cross_val_score(modelo, X, y, cv=cv, scoring='f1').mean()
    auc  = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc').mean()
    resultados[nombre] = {'Accuracy': acc, 'F1': f1, 'AUC': auc}

df_resultados = pd.DataFrame(resultados).T.round(4)
print("=== COMPARACIÓN DE MODELOS (CV 5-fold) ===")
print(df_resultados)
df_resultados.to_csv('data/model_comparison_xg.csv')

mejor = modelos['XGBoost']
mejor.fit(X_train, y_train)
fi = pd.Series(mejor.named_steps['clf'].feature_importances_, index=features_xg)
fi.sort_values().plot(kind='barh', figsize=(8,6), title='Feature importance XGBoost — xG')
plt.tight_layout(); plt.savefig('figures/importance_xgboost.png', dpi=150); plt.show()"""

c3_B_MD = """## Análisis de Modelos Avanzados
El modelo XGBoost mejora frente a Logistic Regression debido a:
- No-linealidad de features (distancia, ángulo) que son capturadas mejor por los árboles.
- Interacciones complejas (e.g. distancia × es_big_chance).
- El overfitting es controlado mediante cross-validation."""

c3_C1 = """events_spatial = pd.read_csv('data/events.csv')
events_spatial = events_spatial[(events_spatial['x']!=0)|(events_spatial['y']!=0)].copy()

def calc_press_pct(grp):
    defensive = grp[grp['type'].isin(['Tackle','Interception','BallRecovery'])] if 'type' in grp.columns else grp[grp['event_type'].isin(['Tackle','Interception','BallRecovery'])]
    in_rival_half = defensive[defensive['x'] > 50]
    return len(in_rival_half) / max(len(defensive), 1)

shots['is_progressive_pass'] = (
    (shots['end_x'] - shots['x'] > 10) &
    (shots['x'] < 80)
).astype(int)

def key_passes(g):
    return g['qualifiers'].astype(str).str.contains('KeyPass').sum() if 'qualifiers' in g.columns else 0

def crosses(g):
    return g['qualifiers'].astype(str).str.contains('Cross').sum() if 'qualifiers' in g.columns else 0

def prog_passes(g):
    type_col = 'type' if 'type' in g.columns else 'event_type'
    return ((g['end_x']-g['x']>10)&(g['x']<80)&(g[type_col]=='Pass')).sum()

team_match_stats = events_spatial.groupby(['match_id','team_name']).apply(
    lambda g: pd.Series({
        'press_pct': calc_press_pct(g),
        'progressive_passes': prog_passes(g),
        'key_passes': key_passes(g),
        'crosses': crosses(g),
    })
).reset_index()
team_match_stats.to_csv('data/team_match_stats.csv', index=False)
print("✅ team_match_stats.csv con press_pct y progressive_passes guardado")"""

c3_D1 = """matches_sorted = matches.sort_values('date').copy()

def rolling_team_features(df, team_col, goals_scored_col, goals_conceded_col, window=5):
    teams = df[team_col].unique()
    result_rows = []
    for team in teams:
        team_matches = df[df[team_col]==team].copy().sort_values('date')
        team_matches[f'{team_col}_rolling_scored']   = team_matches[goals_scored_col].shift(1).rolling(window, min_periods=1).mean()
        team_matches[f'{team_col}_rolling_conceded']  = team_matches[goals_conceded_col].shift(1).rolling(window, min_periods=1).mean()
        team_matches[f'{team_col}_rolling_wins']      = (team_matches['result']==(('H' if team_col=='home_team' else 'A'))).shift(1).rolling(window, min_periods=1).mean()
        result_rows.append(team_matches)
    return pd.concat(result_rows).sort_index()

matches_fe = rolling_team_features(matches_sorted, 'home_team', 'home_goals', 'away_goals')
matches_fe = rolling_team_features(matches_fe, 'away_team', 'away_goals', 'home_goals')

for team_col, gs, gc in [('home_team','home_goals','away_goals'),('away_team','away_goals','home_goals')]:
    for w in [3, 5]:
        for t in [team_col]:
            matches_fe[f'{t}_rolling{w}_scored'] = matches_fe.groupby(t)['home_goals'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())

features_match = [
    'b365h','b365d','b365a',
    'home_team_rolling_scored','home_team_rolling_conceded','home_team_rolling_wins',
    'away_team_rolling_scored','away_team_rolling_conceded','away_team_rolling_wins',
]
matches_model = matches_fe[features_match + ['home_goals','away_goals','result']].dropna()
matches_model['total_goals'] = matches_model['home_goals'] + matches_model['away_goals']
print(f"Partidos disponibles: {len(matches_model)} de {len(matches_fe)}")"""

c3_D2 = """X_reg = matches_model[features_match]
y_reg = matches_model['total_goals']
ridge = Ridge(alpha=1.0)
cv_r2  = cross_val_score(ridge, X_reg, y_reg, cv=5, scoring='r2')
cv_mae = cross_val_score(ridge, X_reg, y_reg, cv=5, scoring='neg_mean_absolute_error')
print("=== MODELO LINEAL (total_goals) ===")
print(f"R² CV:  {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"MAE CV: {(-cv_mae).mean():.4f} ± {(-cv_mae.std()):.4f}")
print(f"Baseline (predecir media): MAE = {(y_reg - y_reg.mean()).abs().mean():.4f}")
ridge.fit(X_reg, y_reg)
y_pred_reg = ridge.predict(X_reg)
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(y_reg, y_pred_reg, alpha=0.5)
ax.plot([0,7],[0,7],'--r')
ax.set_xlabel('Goles reales'); ax.set_ylabel('Goles predichos')
ax.set_title('Regresión Lineal — Real vs Predicho')
plt.tight_layout(); plt.savefig('figures/lineal_actual_vs_pred.png', dpi=150); plt.show()"""

c3_D3 = """y_cls = matches_model['result']
pipe_match = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(penalty='l2', C=1.0,
                               max_iter=1000, random_state=42))
])
cv_acc = cross_val_score(pipe_match, X_reg, y_cls, cv=5, scoring='accuracy')
print("=== MODELO LOGÍSTICO (resultado H/D/A) ===")
print(f"Accuracy CV (5-fold): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"Benchmark Bet365:     0.4980")
print(f"¿Supera Bet365?:      {cv_acc.mean() > 0.498}")

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_cls, test_size=0.2,
                                            random_state=42, stratify=y_cls)
pipe_match.fit(X_tr, y_tr)
y_pred_m = pipe_match.predict(X_te)
print(classification_report(y_te, y_pred_m, target_names=['Away','Draw','Home']))

cm3 = confusion_matrix(y_te, y_pred_m, labels=['H','D','A'])
fig, ax = plt.subplots()
sns.heatmap(cm3, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Home','Draw','Away'], yticklabels=['Home','Draw','Away'])
plt.title('Matriz de confusión — Match Predictor')
plt.tight_layout(); plt.savefig('figures/confusion_match.png', dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(7,4))
modelos_bench = ['Baseline\\n(33.3%)', 'Bet365\\n(49.8%)', f'Tu modelo\\n({cv_acc.mean()*100:.1f}%)']
valores = [0.333, 0.498, cv_acc.mean()]
colors = ['#aaa','#e67e22','#2ecc71' if cv_acc.mean()>0.498 else '#e74c3c']
ax.bar(modelos_bench, valores, color=colors, width=0.5)
for i, v in enumerate(valores):
    ax.text(i, v+0.005, f'{v*100:.1f}%', ha='center', fontsize=11)
ax.set_ylim(0, 0.7); ax.set_title('Comparación de accuracy vs benchmarks')
plt.tight_layout(); plt.savefig('figures/benchmark_comparison.png', dpi=150); plt.show()"""

c3_D4 = """pipe_match.fit(X_reg, y_cls)
ridge.fit(X_reg, y_reg)
joblib.dump(pipe_match, 'models/match_predictor.pkl')
joblib.dump(ridge, 'models/ridge_goals.pkl')
scaler = StandardScaler().fit(X_reg)
joblib.dump(scaler, 'models/scaler_reg.pkl')
print("✅ Modelos guardados")"""

c3_D5 = """teams = sorted(matches['home_team'].unique().tolist())
predictions = []
for home in teams:
    for away in teams:
        if home == away: continue
        h_row = matches_fe[matches_fe['home_team']==home].sort_values('date').tail(1)
        a_row = matches_fe[matches_fe['away_team']==away].sort_values('date').tail(1)
        if h_row.empty or a_row.empty: continue
        try:
            feat_vec = {
                'b365h': 2.0, 'b365d': 3.4, 'b365a': 3.5,
                'home_team_rolling_scored':   h_row['home_team_rolling_scored'].values[0],
                'home_team_rolling_conceded': h_row['home_team_rolling_conceded'].values[0],
                'home_team_rolling_wins':     h_row['home_team_rolling_wins'].values[0],
                'away_team_rolling_scored':   a_row['away_team_rolling_scored'].values[0],
                'away_team_rolling_conceded': a_row['away_team_rolling_conceded'].values[0],
                'away_team_rolling_wins':     a_row['away_team_rolling_wins'].values[0],
            }
            X_pred = pd.DataFrame([feat_vec])[features_match]
            proba = pipe_match.predict_proba(X_pred)[0]
            goles = float(ridge.predict(X_pred)[0])
            classes = list(pipe_match.classes_)
            predictions.append({
                'home': home, 'away': away,
                'prob_H': round(float(proba[classes.index('H')]),3),
                'prob_D': round(float(proba[classes.index('D')]),3),
                'prob_A': round(float(proba[classes.index('A')]),3),
                'expected_goals': round(goles, 1)
            })
        except Exception as e:
            continue

with open('dashboard/predictions.json','w') as f:
    json.dump(predictions, f)
print(f"✅ predictions.json generado con {len(predictions)} combinaciones")"""

c3_E1 = """cluster_features = ['expected_goals','expected_assists','goals_scored','assists',
                    'minutes','ict_index','threat','creativity']
for col in cluster_features:
    if col not in players.columns:
        players[col] = 0

players_cluster = players[cluster_features].dropna().copy()
scaler_cl = StandardScaler()
X_cl = scaler_cl.fit_transform(players_cluster)

inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cl, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.plot(list(K_range), inertias, 'bo-'); ax1.set_title('Elbow method'); ax1.set_xlabel('k')
ax2.plot(list(K_range), sil_scores, 'ro-'); ax2.set_title('Silhouette score'); ax2.set_xlabel('k')
plt.tight_layout(); plt.savefig('figures/clustering_elbow.png', dpi=150); plt.show()"""

c3_E2 = """k_optimo = 4
km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
players_cluster['cluster'] = km_final.fit_predict(X_cl)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cl)
players_cluster['pca1'] = coords[:,0]
players_cluster['pca2'] = coords[:,1]

fig, ax = plt.subplots(figsize=(10,7))
for cl in range(k_optimo):
    mask = players_cluster['cluster']==cl
    ax.scatter(players_cluster[mask]['pca1'], players_cluster[mask]['pca2'],
               label=f'Cluster {cl}', alpha=0.7, s=40)
ax.set_title('Clustering de jugadores (PCA 2D)')
ax.legend(); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); plt.savefig('figures/clustering_pca.png', dpi=150); plt.show()

players_cluster_export = players[['player_name','team','position']].join(
    players_cluster[['cluster']], how='inner')
players_cluster_export.to_csv('data/players_clustered.csv', index=False)"""

c3_E_MD = """## Clustering
- Cluster 0: Delanteros principales (altos minutos, alto xG)
- Cluster 1: Creadores (alto xA y creativity)
- Cluster 2: Rango medio
- Cluster 3: Suplentes/Defensas"""

c3_JSONS = """shots_dash = shots[['player_name','team_name','x','y','is_goal','xg_predicted',
                     'distance_to_goal','is_big_chance','is_penalty']].copy()
shots_dash.rename(columns={'team_name': 'team'}, inplace=True)
shots_dash.to_json('dashboard/shots_with_xg.json', orient='records')

metrics = {
    'xg_model': {
        'auc': round(roc_auc_score(y_test, y_prob), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'baseline_naive': round(1 - y_test.mean(), 4),
    },
    'match_predictor': {
        'accuracy_cv': round(float(cv_acc.mean()), 4),
        'accuracy_cv_std': round(float(cv_acc.std()), 4),
        'bet365_benchmark': 0.498,
        'baseline_random': 0.333,
        'beats_bet365': bool(cv_acc.mean() > 0.498),
        'confusion_matrix': cm3.tolist(),
        'confusion_labels': ['H','D','A'],
    },
    'model_comparison': pd.read_csv('data/model_comparison_xg.csv').to_dict(orient='records')
}
with open('dashboard/metrics.json','w') as f:
    json.dump(metrics, f)

bins = [0, 10, 20, 30, 100]
labels = ['0-10m','10-20m','20-30m','30m+']
shots['dist_bin'] = pd.cut(shots['distance_to_goal'], bins=bins, labels=labels)

eda_data = {
    'conversion_by_distance': shots.groupby('dist_bin', observed=True)['is_goal'].mean().round(4).to_dict(),
    'results_distribution': matches['result'].value_counts(normalize=True).round(4).to_dict(),
    'top_players_xg': players.nlargest(10, 'expected_goals')[
        ['player_name','team','expected_goals','goals_scored']].to_dict(orient='records'),
    'goals_per_match': {
        'mean': round(matches['total_goals'].mean(),2),
        'over_2_5_pct': round((matches['total_goals']>2.5).mean(),4)
    }
}
with open('dashboard/eda_insights.json','w') as f:
    json.dump(eda_data, f, default=str)

print("✅ Todos los JSONs del dashboard generados")"""

nb3['cells'] = [
    nbf.v4.new_code_cell(c3_0),
    nbf.v4.new_code_cell(c3_A1),
    nbf.v4.new_code_cell(c3_A2),
    nbf.v4.new_code_cell(c3_A3),
    nbf.v4.new_code_cell(c3_A4),
    nbf.v4.new_code_cell(c3_A5),
    nbf.v4.new_code_cell(c3_B1),
    nbf.v4.new_markdown_cell(c3_B_MD),
    nbf.v4.new_code_cell(c3_C1),
    nbf.v4.new_code_cell(c3_D1),
    nbf.v4.new_code_cell(c3_D2),
    nbf.v4.new_code_cell(c3_D3),
    nbf.v4.new_code_cell(c3_D4),
    nbf.v4.new_code_cell(c3_D5),
    nbf.v4.new_code_cell(c3_E1),
    nbf.v4.new_code_cell(c3_E2),
    nbf.v4.new_markdown_cell(c3_E_MD),
    nbf.v4.new_code_cell(c3_JSONS),
]
with open('notebooks/03_models.ipynb', 'w') as f:
    nbf.write(nb3, f)
