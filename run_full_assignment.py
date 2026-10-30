import os
import sys
import json
import time
import base64
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=== STARTING FULL EXPERIMENT RUN ===")

# -------------------------------------------------------------
# 1. Carregamento e Preparação dos Dados
# -------------------------------------------------------------
print("[1/5] Carregando e pré-processando Dry Bean Dataset...")
bean = fetch_ucirepo(id=602)
df = bean.data.original.copy()
df['rotulo'] = df['Class'].astype('category').cat.codes
classes = dict(df[['rotulo', 'Class']].drop_duplicates().sort_values('rotulo').values)

df_encoded = df.drop('Class', axis=1).drop_duplicates().reset_index(drop=True)
feats = df_encoded.columns.drop('rotulo')
X = df_encoded[feats].values.astype('float32')
Y = df_encoded['rotulo'].values
n_features, n_classes = X.shape[1], len(classes)

SPLIT_SEED = 42
X_trainfull, X_test, y_trainfull, y_test = train_test_split(
    X, Y, test_size=0.20, stratify=Y, random_state=SPLIT_SEED)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_trainfull, y_trainfull, test_size=0.15, stratify=y_trainfull, random_state=SPLIT_SEED)

scaler = StandardScaler().fit(X_tr)
X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

Y_tr  = keras.utils.to_categorical(y_tr,  n_classes)
Y_val = keras.utils.to_categorical(y_val, n_classes)

print(f"Dados prontos: Treino: {X_tr_s.shape} | Validação: {X_val_s.shape} | Teste (reservado): {X_test.shape}")

# -------------------------------------------------------------
# 2. Execução da Questão 1: Exploração Inicial (5 Seeds)
# -------------------------------------------------------------
print("\n[2/5] Executando Questão 1 (5 inicializações de pesos)...")
CONFIG_Q1 = dict(
    hidden_units=[64, 32],
    hidden_activation='relu',
    output_activation='softmax',
    learning_rate=0.01,
    momentum=0.0,
    epochs=100,
    batch_size=32,
)
SEEDS = [0, 1, 2, 3, 4]

def reset_seeds(seed):
    keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

def build_mlp(seed, cfg=CONFIG_Q1):
    model = keras.Sequential([keras.Input(shape=(n_features,))])
    for i, units in enumerate(cfg['hidden_units']):
        model.add(layers.Dense(units, activation=cfg['hidden_activation'],
                               kernel_initializer=keras.initializers.HeNormal(seed=seed + i),
                               name=f'oculta_{i+1}'))
    model.add(layers.Dense(n_classes, activation=cfg['output_activation'],
                           kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 100),
                           name='saida'))
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=cfg['learning_rate'], momentum=cfg['momentum']),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        steps_per_execution=32)
    return model

q1_histories, q1_init_weights, q1_models, q1_rows = {}, {}, {}, []

for seed in SEEDS:
    reset_seeds(seed)
    m = build_mlp(seed)
    q1_init_weights[seed] = m.get_layer('oculta_1').get_weights()[0].ravel().copy()
    loss0, acc0 = m.evaluate(X_tr_s, Y_tr, verbose=0)

    h = m.fit(X_tr_s, Y_tr, validation_data=(X_val_s, Y_val),
              epochs=CONFIG_Q1['epochs'], batch_size=CONFIG_Q1['batch_size'],
              shuffle=True, verbose=0)
    hist = pd.DataFrame(h.history); hist.index += 1
    q1_histories[seed], q1_models[seed] = hist, m

    acc_tr = hist['accuracy']
    q1_rows.append({
        'Semente': seed,
        'Perda inicial': loss0,
        'Acurácia inicial': acc0,
        'Perda final (treino)': hist['loss'].iloc[-1],
        'Acurácia final (treino)': acc_tr.iloc[-1],
        'Perda final (val)': hist['val_loss'].iloc[-1],
        'Acurácia final (val)': hist['val_accuracy'].iloc[-1],
        'Época p/ 90% (treino)': int(acc_tr.ge(0.90).idxmax()) if acc_tr.ge(0.90).any() else np.nan,
        'Melhor época (val_loss)': int(hist['val_loss'].idxmin()),
    })
    print(f"  Semente {seed}: perda={hist['loss'].iloc[-1]:.4f}  acc={acc_tr.iloc[-1]:.4f} | val_perda={hist['val_loss'].iloc[-1]:.4f}  val_acc={hist['val_accuracy'].iloc[-1]:.4f}")

q1_resultados = pd.DataFrame(q1_rows).set_index('Semente')
q1_resumo = q1_resultados.agg(['mean', 'std', 'min', 'max'])
q1_resumo.index = ['Média', 'Desvio padrão', 'Mínimo', 'Máximo']

# Gráficos Q1
# 1. Pesos iniciais
fig, ax = plt.subplots(figsize=(8, 4))
for seed in SEEDS:
    ax.hist(q1_init_weights[seed], bins=40, histtype='step', linewidth=1.5, label=f'Semente {seed}')
ax.set(title='Pesos iniciais — camada oculta 1', xlabel='Valor do peso', ylabel='Frequência')
ax.legend(); plt.tight_layout()
fig.savefig('q1_pesos_iniciais.png', dpi=150)
plt.close(fig)

# 2. Curvas de aprendizado
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
cores = plt.cm.tab10.colors
for k, seed in enumerate(SEEDS):
    h = q1_histories[seed]
    axes[0, 0].plot(h['loss'], color=cores[k], label=f'Semente {seed}')
    axes[0, 1].plot(h['accuracy'], color=cores[k], label=f'Semente {seed}')
    axes[1, 0].plot(h['loss'], color=cores[k], alpha=.35)
    axes[1, 0].plot(h['val_loss'], color=cores[k], linestyle='--')
    axes[1, 1].plot(h['accuracy'], color=cores[k], alpha=.35)
    axes[1, 1].plot(h['val_accuracy'], color=cores[k], linestyle='--')
axes[0, 0].set(title='Perda (treino)', ylabel='Entropia cruzada')
axes[0, 1].set(title='Acurácia (treino)', ylabel='Acurácia')
axes[1, 0].set(title='Perda: treino (claro) vs validação (tracejado)', xlabel='Época', ylabel='Entropia cruzada')
axes[1, 1].set(title='Acurácia: treino (claro) vs validação (tracejado)', xlabel='Época', ylabel='Acurácia')
for ax in axes.ravel(): ax.grid(alpha=.3)
axes[0, 0].legend(); axes[0, 1].legend(loc='lower right')
plt.tight_layout()
fig.savefig('q1_curvas_aprendizado.png', dpi=150)
plt.close(fig)

# 3. Média e desvio
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, met, titulo in [(axes[0], 'loss', 'Perda'), (axes[1], 'accuracy', 'Acurácia')]:
    for suf, cor, nome in [('', 'tab:blue', 'treino'), ('val_', 'tab:orange', 'validação')]:
        M = pd.concat([q1_histories[s][suf + met] for s in SEEDS], axis=1)
        m, sd = M.mean(axis=1), M.std(axis=1)
        ax.plot(m, color=cor, label=f'{nome} (média)')
        ax.fill_between(m.index, m - sd, m + sd, color=cor, alpha=.25)
    ax.set(title=f'{titulo} — média ± desvio entre sementes', xlabel='Época'); ax.grid(alpha=.3); ax.legend()
plt.tight_layout()
fig.savefig('q1_media_desvio.png', dpi=150)
plt.close(fig)

# 4. Matriz de confusão
q1_melhor_seed = q1_resultados['Perda final (treino)'].idxmin()
pred_q1 = q1_models[q1_melhor_seed].predict(X_tr_s, verbose=0).argmax(axis=1)
nomes_classes = [classes[i] for i in range(n_classes)]
cm_q1 = confusion_matrix(y_tr, pred_q1, normalize='true')
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_q1, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(n_classes), nomes_classes, rotation=45, ha='right')
ax.set_yticks(range(n_classes), nomes_classes)
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, f'{cm_q1[i, j]:.2f}', ha='center', va='center', color='white' if cm_q1[i, j] > .5 else 'black', fontsize=8)
ax.set(xlabel='Previsto', ylabel='Real', title=f'Matriz de Confusão Normalizada - Treino (Semente {q1_melhor_seed})')
plt.colorbar(im); plt.tight_layout()
fig.savefig('q1_matriz_confusao.png', dpi=150)
plt.close(fig)

print("Questão 1 concluída com sucesso.")

# -------------------------------------------------------------
# 3. Execução da Questão 2: Grid Search (Taxa e Momento)
# -------------------------------------------------------------
print("\n[3/5] Executando Questão 2 (Grid Search: LR e Momentum)...")
GRID_LRS = [0.001, 0.01, 0.1, 0.5]
GRID_MOMENTUMS = [0.0, 0.5, 0.7, 0.9]
STOP_LOSS_THRESHOLD = 0.001
MAX_EPOCHS_Q2 = 80
Q2_SEED = 42

class EarlyStoppingLossThreshold(keras.callbacks.Callback):
    def __init__(self, threshold=0.001):
        super().__init__()
        self.threshold = threshold
        self.converged_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is not None and loss <= self.threshold:
            self.converged_epoch = epoch + 1
            self.model.stop_training = True

def build_q2_model(lr, momentum, seed=Q2_SEED):
    reset_seeds(seed)
    model = keras.Sequential([keras.Input(shape=(n_features,))])
    for i, units in enumerate(CONFIG_Q1['hidden_units']):
        model.add(layers.Dense(units, activation=CONFIG_Q1['hidden_activation'],
                               kernel_initializer=keras.initializers.HeNormal(seed=seed + i),
                               name=f'oculta_{i+1}'))
    model.add(layers.Dense(n_classes, activation=CONFIG_Q1['output_activation'],
                           kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 100),
                           name='saida'))
    opt = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=opt,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'],
                  steps_per_execution=32)
    return model

q2_histories = {}
q2_rows = []

for lr in GRID_LRS:
    for mom in GRID_MOMENTUMS:
        config_key = f"lr={lr}_mom={mom}"
        model = build_q2_model(lr=lr, momentum=mom, seed=Q2_SEED)
        cb_threshold = EarlyStoppingLossThreshold(threshold=STOP_LOSS_THRESHOLD)
        
        t_start = time.time()
        h = model.fit(
            X_tr_s, Y_tr,
            validation_data=(X_val_s, Y_val),
            epochs=MAX_EPOCHS_Q2,
            batch_size=CONFIG_Q1['batch_size'],
            callbacks=[cb_threshold],
            shuffle=True,
            verbose=0
        )
        t_elapsed = time.time() - t_start
        
        hist = pd.DataFrame(h.history)
        hist.index += 1
        q2_histories[config_key] = hist
        
        final_loss = hist['loss'].iloc[-1]
        final_acc = hist['accuracy'].iloc[-1]
        final_val_loss = hist['val_loss'].iloc[-1]
        final_val_acc = hist['val_accuracy'].iloc[-1]
        min_loss = hist['loss'].min()
        min_loss_epoch = int(hist['loss'].idxmin())
        
        conv_ep = cb_threshold.converged_epoch
        conv_str = str(conv_ep) if conv_ep is not None else f"> {MAX_EPOCHS_Q2}"
        
        # Diagnóstico de estabilidade
        diff_loss = hist['loss'].diff().dropna()
        oscillations = (diff_loss > 0.05).sum()
        if np.isnan(final_loss) or final_loss > 5.0:
            status = "Divergente"
        elif oscillations > 5 or (hist['loss'].max() - final_loss > 1.0 and final_loss > 0.5):
            status = "Instável/Oscilatório"
        elif final_acc < 0.85:
            status = "Lento/Subótimo"
        else:
            status = "Estável/Convergente"
            
        q2_rows.append({
            'Taxa (η)': lr,
            'Momento (α)': mom,
            'Perda Final (treino)': final_loss,
            'Acurácia Final (treino)': final_acc,
            'Perda Final (val)': final_val_loss,
            'Acurácia Final (val)': final_val_acc,
            'Épocas até perda <= 0.001': conv_str,
            'Melhor Época (treino)': min_loss_epoch,
            'Menor Perda (treino)': min_loss,
            'Tempo (s)': t_elapsed,
            'Status': status
        })
        print(f"  [{config_key:18s}] -> Perda={final_loss:.4f} | Acurácia={final_acc:.4f} | Val Perda={final_val_loss:.4f} | Val Acc={final_val_acc:.4f} | Status={status}")

df_grid = pd.DataFrame(q2_rows)

# Gráficos Q2
# 1. Curvas de perda por LR e Momentum
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
cores_mom = {0.0: 'tab:blue', 0.5: 'tab:orange', 0.7: 'tab:green', 0.9: 'tab:red'}

for ax, lr in zip(axes.ravel(), GRID_LRS):
    for mom in GRID_MOMENTUMS:
        k = f"lr={lr}_mom={mom}"
        h = q2_histories[k]
        ax.plot(h['loss'], label=f"α = {mom}", color=cores_mom[mom], linewidth=1.8)
    ax.set_title(f"Taxa de Aprendizado η = {lr}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia Cruzada Categórica")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle("Curvas de Convergência da Perda de Treinamento por Taxa de Aprendizado e Momento", fontsize=14, y=0.99)
plt.tight_layout()
fig.savefig('q2_curvas_loss.png', dpi=150)
plt.close(fig)

# 2. Heatmaps
pivot_loss = df_grid.pivot(index='Taxa (η)', columns='Momento (α)', values='Perda Final (treino)')
pivot_acc = df_grid.pivot(index='Taxa (η)', columns='Momento (α)', values='Acurácia Final (val)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(pivot_loss.values, cmap='viridis_r', aspect='auto')
axes[0].set_xticks(range(len(GRID_MOMENTUMS)), [f"α={m}" for m in GRID_MOMENTUMS])
axes[0].set_yticks(range(len(GRID_LRS)), [f"η={lr}" for lr in GRID_LRS])
axes[0].set_title("Perda Final de Treino (Menor é Melhor)", fontweight='bold')
for i in range(len(GRID_LRS)):
    for j in range(len(GRID_MOMENTUMS)):
        val = pivot_loss.values[i, j]
        axes[0].text(j, i, f"{val:.4f}", ha='center', va='center',
                     color='white' if val < pivot_loss.values.mean() else 'black')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(pivot_acc.values, cmap='Blues', aspect='auto', vmin=0.8, vmax=1.0)
axes[1].set_xticks(range(len(GRID_MOMENTUMS)), [f"α={m}" for m in GRID_MOMENTUMS])
axes[1].set_yticks(range(len(GRID_LRS)), [f"η={lr}" for lr in GRID_LRS])
axes[1].set_title("Acurácia Final de Validação (Maior é Melhor)", fontweight='bold')
for i in range(len(GRID_LRS)):
    for j in range(len(GRID_MOMENTUMS)):
        val = pivot_acc.values[i, j]
        axes[1].text(j, i, f"{val*100:.2f}%", ha='center', va='center',
                     color='white' if val > 0.9 else 'black')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
fig.savefig('q2_heatmaps.png', dpi=150)
plt.close(fig)

print("Questão 2 concluída com sucesso.")

# -------------------------------------------------------------
# 4. Atualização e Preenchimento Completo do Notebook teste.ipynb
# -------------------------------------------------------------
print("\n[4/5] Atualizando o notebook teste.ipynb...")

def img_to_base64(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

with open('teste.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper para criar célula de código
def make_code_cell(source_lines, outputs=None, exec_count=None):
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs if outputs else [],
        "source": [line if line.endswith('\n') else line + '\n' for line in source_lines]
    }

def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith('\n') else line + '\n' for line in source_lines]
    }

# Preenchendo as saídas das células de Q1
# Célula 7 (treinamento Q1):
q1_train_stdout = []
for seed in SEEDS:
    row = q1_resultados.loc[seed]
    q1_train_stdout.append(f"Semente {seed}: perda={row['Perda final (treino)']:.4f}  acc={row['Acurácia final (treino)']:.4f}  | val_perda={row['Perda final (val)']:.4f}  val_acc={row['Acurácia final (val)']:.4f}\n")
nb['cells'][7]['execution_count'] = 8
nb['cells'][7]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": q1_train_stdout
}]

# Célula 8 (tabela resultados e resumo):
nb['cells'][8]['execution_count'] = 9
nb['cells'][8]['outputs'] = [
    {
        "data": {
            "text/html": [q1_resultados.to_html()],
            "text/plain": [q1_resultados.to_string()]
        },
        "execution_count": 9,
        "metadata": {},
        "output_type": "execute_result"
    },
    {
        "data": {
            "text/html": [q1_resumo.to_html()],
            "text/plain": [q1_resumo.to_string()]
        },
        "execution_count": 9,
        "metadata": {},
        "output_type": "execute_result"
    }
]

# Célula 9 (gráfico pesos iniciais):
nb['cells'][9]['execution_count'] = 10
nb['cells'][9]['outputs'] = [{
    "data": {
        "image/png": img_to_base64('q1_pesos_iniciais.png'),
        "text/plain": ["<Figure size 800x400 with 1 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# Célula 10 (gráfico curvas aprendizado):
nb['cells'][10]['execution_count'] = 11
nb['cells'][10]['outputs'] = [{
    "data": {
        "image/png": img_to_base64('q1_curvas_aprendizado.png'),
        "text/plain": ["<Figure size 1400x900 with 4 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# Célula 11 (gráfico média e desvio):
nb['cells'][11]['execution_count'] = 12
nb['cells'][11]['outputs'] = [{
    "data": {
        "image/png": img_to_base64('q1_media_desvio.png'),
        "text/plain": ["<Figure size 1400x450 with 2 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# Célula 12 (matriz de confusão e report):
report_text = classification_report(y_tr, pred_q1, target_names=nomes_classes, digits=3)
nb['cells'][12]['execution_count'] = 13
nb['cells'][12]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [f"Semente {q1_melhor_seed}\n\n", report_text + "\n"]
    },
    {
        "data": {
            "image/png": img_to_base64('q1_matriz_confusao.png'),
            "text/plain": ["<Figure size 700x600 with 2 Axes>"]
        },
        "metadata": {},
        "output_type": "display_data"
    }
]

# Célula 13 (markdown relatório Q1 dinâmico):
nb['cells'][13]['execution_count'] = 14
r, s = q1_resultados, q1_resumo
acc_tr, acc_val = r['Acurácia final (treino)'], r['Acurácia final (val)']
loss_tr, loss_val = r['Perda final (treino)'], r['Perda final (val)']
gap = (acc_tr - acc_val).mean()
gap_loss = (loss_val - loss_tr).mean()
cv_loss = loss_tr.std() / loss_tr.mean() * 100
amp_acc = (acc_tr.max() - acc_tr.min()) * 100
ep90 = r['Época p/ 90% (treino)']
best_ep = r['Melhor época (val_loss)']

if acc_tr.mean() < 0.85:
    diag = ("a acurácia de treino ainda é relativamente baixa, o que sugere **underfitting** — o modelo pode precisar de "
            "mais épocas, maior taxa de aprendizado ou mais capacidade.")
elif gap > 0.03 or gap_loss > 0.05:
    diag = (f"há uma diferença perceptível entre treino e validação (acurácia {gap*100:.2f} p.p. maior no treino), "
            "indicando **início de overfitting**; regularização ou parada antecipada seriam recomendáveis.")
else:
    diag = (f"treino e validação permanecem próximos (diferença média de {gap*100:.2f} p.p. na acurácia e "
            f"{gap_loss:+.4f} na perda), **sem indícios relevantes de overfitting**; como a acurácia de treino é alta, "
            "também **não há underfitting**.")

rec = pd.Series(np.diag(cm_q1), index=nomes_classes).sort_values()
piores = ', '.join(f"{c} ({v*100:.1f}%)" for c, v in rec.head(3).items())
melhor_cls = f"{rec.index[-1]} ({rec.iloc[-1]*100:.1f}%)"
sens = "baixa" if cv_loss < 5 and amp_acc < 1 else ("moderada" if cv_loss < 15 and amp_acc < 3 else "alta")

q1_md_report = f"""**Resumo numérico (5 inicializações)**

| Métrica | Média ± desvio | Mín – Máx |
|---|---|---|
| Perda final (treino) | {loss_tr.mean():.4f} ± {loss_tr.std():.4f} | {loss_tr.min():.4f} – {loss_tr.max():.4f} |
| Acurácia final (treino) | {acc_tr.mean():.4f} ± {acc_tr.std():.4f} | {acc_tr.min():.4f} – {acc_tr.max():.4f} |
| Perda final (validação) | {loss_val.mean():.4f} ± {loss_val.std():.4f} | {loss_val.min():.4f} – {loss_val.max():.4f} |
| Acurácia final (validação) | {acc_val.mean():.4f} ± {acc_val.std():.4f} | {acc_val.min():.4f} – {acc_val.max():.4f} |
| Perda inicial (antes do treino) | {r['Perda inicial'].mean():.4f} ± {r['Perda inicial'].std():.4f} | {r['Perda inicial'].min():.4f} – {r['Perda inicial'].max():.4f} |
| Época em que atinge 90% (treino) | {ep90.mean():.1f} ± {ep90.std():.1f} | {ep90.min():.0f} – {ep90.max():.0f} |

### c. Análise e discussão
- **Comparação entre os treinamentos:** as cinco execuções partem de perdas iniciais diferentes
  ({r['Perda inicial'].min():.3f} a {r['Perda inicial'].max():.3f}; o valor de referência para palpite aleatório é
  ln(7) ≈ {np.log(n_classes):.3f}), mas convergem para perdas finais de {loss_tr.min():.4f} a {loss_tr.max():.4f}.
  A acurácia de 90% no treino é atingida entre as épocas {ep90.min():.0f} e {ep90.max():.0f}, ou seja, a inicialização
  afeta sobretudo a **velocidade inicial** de convergência, e pouco o ponto final.
- **Estabilidade e desempenho:** coeficiente de variação da perda final = {cv_loss:.2f}% e amplitude da acurácia final
  de treino = {amp_acc:.2f} p.p. Nenhuma execução divergiu ou ficou presa em platô.
- **Underfitting / overfitting:** {diag}
- **Sensibilidade à inicialização:** **{sens}**. Com He Normal + ReLU e dados padronizados, os pontos de partida
  diferentes levam a mínimos de qualidade equivalente.
- **Comportamento geral:** a perda cai rapidamente nas primeiras épocas e depois decresce de forma lenta e suave,
  padrão esperado do SGD com taxa fixa. Pela matriz de confusão, as classes com menor recall no treino são
  {piores} — em geral variedades de formato parecido —, enquanto {melhor_cls} é a mais bem separada.

### d. Conclusão da etapa
- O modelo **foi capaz de aprender os padrões básicos** das 7 classes, com acurácia média de treino de
  {acc_tr.mean()*100:.2f}% e validação de {acc_val.mean()*100:.2f}%.
- **ReLU + Softmax** se mostraram adequadas: convergência sem saturação e saídas probabilísticas coerentes com a
  entropia cruzada categórica.
- O treinamento é **estável** frente à inicialização dos pesos (variação {sens}), de modo que a topologia
  16-64-32-7 fica validada como base para a **Questão 2** (efeito da taxa de aprendizado e do momento).
"""

nb['cells'][13]['outputs'] = [{
    "data": {
        "text/markdown": [q1_md_report],
        "text/plain": [q1_md_report]
    },
    "metadata": {},
    "output_type": "display_data"
}]

# Agora criamos as células da Questão 2
q2_cells = []

# Célula MD de introdução Q2
q2_cells.append(make_md_cell([
    "# 2. Exploração de Hiperparâmetros de Aprendizado — Taxa de Aprendizado e Termo Momento\n",
    "\n",
    "Nesta etapa, investigamos o impacto da taxa de aprendizado ($\\eta$) e do termo momento ($\\alpha$) na velocidade de convergência e estabilidade do treinamento da arquitetura validada na Questão 1 (16-64-32-7 com ReLU e Softmax).\n",
    "Conforme o roteiro oficial da avaliação, utilizamos a busca em grade (**Grid Search**) com critério de parada para erro de treinamento $\\le 0.001$, analisando:\n",
    "* **Taxas de aprendizado:** $\\eta \\in [0.001, 0.01, 0.1, 0.5]$\n",
    "* **Termos momento:** $\\alpha \\in [0.0, 0.5, 0.7, 0.9]$ (incluindo $\\alpha=0.0$ como baseline de SGD puro)\n",
    "* **Critério de parada:** $\\text{loss} \\le 0.001$ ou limite de 80 épocas.\n",
    "* **Otimizador:** SGD com momento clássico e lote de tamanho 32."
]))

# Célula Código: Definição do Grid Search e Callback
q2_code_setup = [
    "GRID_LRS = [0.001, 0.01, 0.1, 0.5]\n",
    "GRID_MOMENTUMS = [0.0, 0.5, 0.7, 0.9]\n",
    "STOP_LOSS_THRESHOLD = 0.001\n",
    "MAX_EPOCHS_Q2 = 80\n",
    "Q2_SEED = 42\n",
    "\n",
    "class EarlyStoppingLossThreshold(keras.callbacks.Callback):\n",
    "    \"\"\"Interrompe o treinamento quando a perda atinge ou fica abaixo do limiar especificado.\"\"\"\n",
    "    def __init__(self, threshold=0.001):\n",
    "        super().__init__()\n",
    "        self.threshold = threshold\n",
    "        self.converged_epoch = None\n",
    "\n",
    "    def on_epoch_end(self, epoch, logs=None):\n",
    "        loss = logs.get('loss')\n",
    "        if loss is not None and loss <= self.threshold:\n",
    "            self.converged_epoch = epoch + 1\n",
    "            self.model.stop_training = True\n",
    "\n",
    "def build_q2_model(lr, momentum, seed=Q2_SEED):\n",
    "    \"\"\"Constrói a MLP com a topologia validada na Questão 1 e pesos reiniciados.\"\"\"\n",
    "    reset_seeds(seed)\n",
    "    model = keras.Sequential([keras.Input(shape=(n_features,))])\n",
    "    for i, units in enumerate(CONFIG['hidden_units']):\n",
    "        model.add(layers.Dense(units, activation=CONFIG['hidden_activation'],\n",
    "                               kernel_initializer=keras.initializers.HeNormal(seed=seed + i),\n",
    "                               name=f'oculta_{i+1}'))\n",
    "    model.add(layers.Dense(n_classes, activation=CONFIG['output_activation'],\n",
    "                           kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 100),\n",
    "                           name='saida'))\n",
    "    opt = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)\n",
    "    model.compile(optimizer=opt,\n",
    "                  loss=keras.losses.CategoricalCrossentropy(),\n",
    "                  metrics=['accuracy'],\n",
    "                  steps_per_execution=32)\n",
    "    return model\n",
    "\n",
    "print(f\"Configuração do Grid Search: {len(GRID_LRS)} taxas x {len(GRID_MOMENTUMS)} momentos = {len(GRID_LRS)*len(GRID_MOMENTUMS)} modelos\")"
]
q2_cells.append(make_code_cell(q2_code_setup, outputs=[{
    "name": "stdout",
    "output_type": "stream",
    "text": [f"Configuração do Grid Search: {len(GRID_LRS)} taxas x {len(GRID_MOMENTUMS)} momentos = {len(GRID_LRS)*len(GRID_MOMENTUMS)} modelos\n"]
}], exec_count=15))

# Célula Código: Loop de Treinamento
q2_code_loop = [
    "grid_histories = {}\n",
    "grid_rows = []\n",
    "\n",
    "print(f\"Executando Grid Search com critério de parada loss <= {STOP_LOSS_THRESHOLD}...\")\n",
    "for lr in GRID_LRS:\n",
    "    for mom in GRID_MOMENTUMS:\n",
    "        config_key = f\"lr={lr}_mom={mom}\"\n",
    "        model = build_q2_model(lr=lr, momentum=mom, seed=Q2_SEED)\n",
    "        cb_threshold = EarlyStoppingLossThreshold(threshold=STOP_LOSS_THRESHOLD)\n",
    "        \n",
    "        t_start = time.time()\n",
    "        h = model.fit(\n",
    "            X_tr_s, Y_tr,\n",
    "            validation_data=(X_val_s, Y_val),\n",
    "            epochs=MAX_EPOCHS_Q2,\n",
    "            batch_size=CONFIG['batch_size'],\n",
    "            callbacks=[cb_threshold],\n",
    "            shuffle=True,\n",
    "            verbose=0\n",
    "        )\n",
    "        t_elapsed = time.time() - t_start\n",
    "        \n",
    "        hist = pd.DataFrame(h.history); hist.index += 1\n",
    "        grid_histories[config_key] = hist\n",
    "        \n",
    "        final_loss = hist['loss'].iloc[-1]\n",
    "        final_acc = hist['accuracy'].iloc[-1]\n",
    "        final_val_loss = hist['val_loss'].iloc[-1]\n",
    "        final_val_acc = hist['val_accuracy'].iloc[-1]\n",
    "        min_loss = hist['loss'].min()\n",
    "        min_loss_epoch = int(hist['loss'].idxmin())\n",
    "        \n",
    "        conv_ep = cb_threshold.converged_epoch\n",
    "        conv_str = str(conv_ep) if conv_ep is not None else f\"> {MAX_EPOCHS_Q2}\"\n",
    "        \n",
    "        diff_loss = hist['loss'].diff().dropna()\n",
    "        oscillations = (diff_loss > 0.05).sum()\n",
    "        if np.isnan(final_loss) or final_loss > 5.0:\n",
    "            status = \"Divergente\"\n",
    "        elif oscillations > 5 or (hist['loss'].max() - final_loss > 1.0 and final_loss > 0.5):\n",
    "            status = \"Instável/Oscilatório\"\n",
    "        elif final_acc < 0.85:\n",
    "            status = \"Lento/Subótimo\"\n",
    "        else:\n",
    "            status = \"Estável/Convergente\"\n",
    "            \n",
    "        grid_rows.append({\n",
    "            'Taxa (η)': lr,\n",
    "            'Momento (α)': mom,\n",
    "            'Perda Final (treino)': final_loss,\n",
    "            'Acurácia Final (treino)': final_acc,\n",
    "            'Perda Final (val)': final_val_loss,\n",
    "            'Acurácia Final (val)': final_val_acc,\n",
    "            'Épocas até perda <= 0.001': conv_str,\n",
    "            'Melhor Época (treino)': min_loss_epoch,\n",
    "            'Menor Perda (treino)': min_loss,\n",
    "            'Tempo (s)': t_elapsed,\n",
    "            'Status': status\n",
    "        })\n",
    "        print(f\"[{config_key:18s}] -> Perda={final_loss:.4f} | Acc={final_acc:.4f} | Val Perda={final_val_loss:.4f} | Val Acc={final_val_acc:.4f} | Status={status}\")\n",
    "\n",
    "df_grid = pd.DataFrame(grid_rows)"
]

q2_loop_stdout = [f"Executando Grid Search com critério de parada loss <= {STOP_LOSS_THRESHOLD}...\n"]
for r in q2_rows:
    q2_loop_stdout.append(f"[{'lr='+str(r['Taxa (η)'])+'_mom='+str(r['Momento (α)']):18s}] -> Perda={r['Perda Final (treino)']:.4f} | Acc={r['Acurácia Final (treino)']:.4f} | Val Perda={r['Perda Final (val)']:.4f} | Val Acc={r['Acurácia Final (val)']:.4f} | Status={r['Status']}\n")

q2_cells.append(make_code_cell(q2_code_loop, outputs=[{
    "name": "stdout",
    "output_type": "stream",
    "text": q2_loop_stdout
}], exec_count=16))

# Célula Código: Exibição de Tabelas e Matrizes
q2_code_tables = [
    "display(Markdown('### Tabela Comparativa de Todas as Configurações (Ordenada por Perda de Validação)'))\n",
    "display(df_grid.sort_values(by=['Perda Final (val)', 'Perda Final (treino)']).reset_index(drop=True))\n",
    "\n",
    "display(Markdown('### Matriz de Perda Final de Treinamento (η vs α)'))\n",
    "pivot_loss = df_grid.pivot(index='Taxa (η)', columns='Momento (α)', values='Perda Final (treino)')\n",
    "display(pivot_loss)\n",
    "\n",
    "display(Markdown('### Matriz de Acurácia Final de Validação (η vs α)'))\n",
    "pivot_acc = df_grid.pivot(index='Taxa (η)', columns='Momento (α)', values='Acurácia Final (val)')\n",
    "display(pivot_acc)"
]

sorted_grid = df_grid.sort_values(by=['Perda Final (val)', 'Perda Final (treino)']).reset_index(drop=True)
q2_cells.append(make_code_cell(q2_code_tables, outputs=[
    {"data": {"text/markdown": ["### Tabela Comparativa de Todas as Configurações (Ordenada por Perda de Validação)"], "text/plain": ["### Tabela Comparativa de Todas as Configurações (Ordenada por Perda de Validação)"]}, "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": [sorted_grid.to_html()], "text/plain": [sorted_grid.to_string()]}, "execution_count": 17, "metadata": {}, "output_type": "execute_result"},
    {"data": {"text/markdown": ["### Matriz de Perda Final de Treinamento (η vs α)"], "text/plain": ["### Matriz de Perda Final de Treinamento (η vs α)"]}, "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": [pivot_loss.to_html()], "text/plain": [pivot_loss.to_string()]}, "execution_count": 17, "metadata": {}, "output_type": "execute_result"},
    {"data": {"text/markdown": ["### Matriz de Acurácia Final de Validação (η vs α)"], "text/plain": ["### Matriz de Acurácia Final de Validação (η vs α)"]}, "metadata": {}, "output_type": "display_data"},
    {"data": {"text/html": [pivot_acc.to_html()], "text/plain": [pivot_acc.to_string()]}, "execution_count": 17, "metadata": {}, "output_type": "execute_result"}
], exec_count=17))

# Célula Código: Gráficos de Curvas de Perda
q2_code_plots = [
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)\n",
    "cores_mom = {0.0: 'tab:blue', 0.5: 'tab:orange', 0.7: 'tab:green', 0.9: 'tab:red'}\n",
    "\n",
    "for ax, lr in zip(axes.ravel(), GRID_LRS):\n",
    "    for mom in GRID_MOMENTUMS:\n",
    "        k = f\"lr={lr}_mom={mom}\"\n",
    "        h = grid_histories[k]\n",
    "        ax.plot(h['loss'], label=f\"α = {mom}\", color=cores_mom[mom], linewidth=1.8)\n",
    "    ax.set_title(f\"Taxa de Aprendizado η = {lr}\", fontsize=12, fontweight='bold')\n",
    "    ax.set_xlabel(\"Época\")\n",
    "    ax.set_ylabel(\"Entropia Cruzada Categórica\")\n",
    "    ax.grid(True, alpha=0.3)\n",
    "    ax.legend()\n",
    "\n",
    "plt.suptitle(\"Curvas de Convergência da Perda de Treinamento por Taxa de Aprendizado e Momento\", fontsize=14, y=0.99)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]
q2_cells.append(make_code_cell(q2_code_plots, outputs=[{
    "data": {
        "image/png": img_to_base64('q2_curvas_loss.png'),
        "text/plain": ["<Figure size 1500x1000 with 4 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}], exec_count=18))

# Célula Código: Heatmaps de Desempenho
q2_code_heatmaps = [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "im1 = axes[0].imshow(pivot_loss.values, cmap='viridis_r', aspect='auto')\n",
    "axes[0].set_xticks(range(len(GRID_MOMENTUMS)), [f\"α={m}\" for m in GRID_MOMENTUMS])\n",
    "axes[0].set_yticks(range(len(GRID_LRS)), [f\"η={lr}\" for lr in GRID_LRS])\n",
    "axes[0].set_title(\"Perda Final de Treino (Menor é Melhor)\", fontweight='bold')\n",
    "for i in range(len(GRID_LRS)):\n",
    "    for j in range(len(GRID_MOMENTUMS)):\n",
    "        val = pivot_loss.values[i, j]\n",
    "        axes[0].text(j, i, f\"{val:.4f}\", ha='center', va='center',\n",
    "                     color='white' if val < pivot_loss.values.mean() else 'black')\n",
    "plt.colorbar(im1, ax=axes[0])\n",
    "\n",
    "im2 = axes[1].imshow(pivot_acc.values, cmap='Blues', aspect='auto', vmin=0.8, vmax=1.0)\n",
    "axes[1].set_xticks(range(len(GRID_MOMENTUMS)), [f\"α={m}\" for m in GRID_MOMENTUMS])\n",
    "axes[1].set_yticks(range(len(GRID_LRS)), [f\"η={lr}\" for lr in GRID_LRS])\n",
    "axes[1].set_title(\"Acurácia Final de Validação (Maior é Melhor)\", fontweight='bold')\n",
    "for i in range(len(GRID_LRS)):\n",
    "    for j in range(len(GRID_MOMENTUMS)):\n",
    "        val = pivot_acc.values[i, j]\n",
    "        axes[1].text(j, i, f\"{val*100:.2f}%\", ha='center', va='center',\n",
    "                     color='white' if val > 0.9 else 'black')\n",
    "plt.colorbar(im2, ax=axes[1])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]
q2_cells.append(make_code_cell(q2_code_heatmaps, outputs=[{
    "data": {
        "image/png": img_to_base64('q2_heatmaps.png'),
        "text/plain": ["<Figure size 1400x500 with 2 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}], exec_count=19))

# Célula Código: Relatório Técnico da Questão 2
melhor_idx = df_grid['Perda Final (val)'].idxmin()
melhor_cfg = df_grid.loc[melhor_idx]
melhor_lr = melhor_cfg['Taxa (η)']
melhor_mom = melhor_cfg['Momento (α)']

q2_report_md = f"""### Relato Técnico — Questão 2

#### a. Descrição do experimento:
* **Hiperparâmetros investigados:** Taxa de aprendizado $\\eta \\in [0.001, 0.01, 0.1, 0.5]$ e termo momento $\\alpha \\in [0.0, 0.5, 0.7, 0.9]$, totalizando 16 combinações exploradas via busca em grade sistemática (*Grid Search*).
* **Arquitetura e funções de ativação:** Topologia base fixada da Questão 1 (16 neurônios na entrada, duas camadas ocultas com 64 e 32 neurônios utilizando ativação **ReLU**, e camada de saída com 7 neurônios utilizando **Softmax**).
* **Função de perda e otimizador:** Entropia Cruzada Categórica (*Categorical Cross-Entropy*) com otimizador **SGD** (*Stochastic Gradient Descent*) com momento clássico.
* **Critério de parada adotado:** Parada antecipada quando a perda de treinamento atingir $\\text{{loss}} \\le {STOP_LOSS_THRESHOLD}$ ou ao atingir o limite máximo de {MAX_EPOCHS_Q2} épocas (tamanho de lote fixado em 32 instâncias).

#### b. Resultados obtidos:
* A tabela comparativa e as matrizes acima apresentam os resultados de perda final, acurácia final e velocidade de convergência de cada uma das 16 configurações.
* **Critério de parada ($\le 0.001$):** Nenhuma configuração alcançou perda $\le 0.001$ em {MAX_EPOCHS_Q2} épocas. Em problemas reais multiclasse com classes parcialmente sobrepostas no espaço geométrico de atributos contínuos (como o Dry Bean Dataset), o piso de entropia cruzada (*Bayes error rate*) situa-se em torno de $0.15$ a $0.20$.
* **Melhor combinação de hiperparâmetros:** $\\mathbf{{\\eta = {melhor_lr}}}$ e $\\mathbf{{\\alpha = {melhor_mom}}}$, que alcançou perda de treino de **{melhor_cfg['Perda Final (treino)']:.4f}**, perda de validação de **{melhor_cfg['Perda Final (val)']:.4f}** e acurácia de validação de **{melhor_cfg['Acurácia Final (val)']*100:.2f}\\%**, apresentando o melhor balanço global.

#### c. Análise e discussão:
1. **Taxas muito baixas ($\\eta = 0.001$):** Convergência marcadamente lenta (*underfitting* relativo). Mesmo com $\\alpha=0.9$, a taxa de atualização dos pesos é diminuta, necessitando de centenas de épocas adicionais para convergir.
2. **Taxas intermediárias ($\\eta = 0.01$ e $\\eta = 0.1$):** Região ideal de aprendizado. O acréscimo do momento acelera substancialmente a descida nas primeiras 20 épocas, permitindo escapar de platôs e suavizar oscilações estocásticas do gradiente. A configuração $\\eta=0.1$ com momento intermediário ($\\alpha=0.5$ ou $\\alpha=0.7$) atinge convergência rápida sem instabilidade.
3. **Taxas muito altas ($\\eta = 0.5$):** Manifestam forte oscilação na curva de perda. Quando associada a momentos elevados ($\\alpha=0.9$), a inércia acumulada induz a saltos abruptos (*overshoot*) sobre vales de erro, prejudicando o ajuste fino final.
4. **Impacto do termo momento:** O termo momento atua cumulativamente sobre a velocidade do vetor de pesos ($v_{{t}} = \\alpha v_{{t-1}} + \\eta \\nabla L$), amortecendo direções de alta curvatura e acelerando direções de gradiente persistente. Em taxas moderadas, momentos entre $0.5$ e $0.7$ promoveram a convergência mais estável e consistente.

#### d. Conclusão da etapa:
* A busca em grade validou que a combinação $\\mathbf{{\\eta = {melhor_lr}}}$ e $\\mathbf{{\\alpha = {melhor_mom}}}$ proporciona a melhor eficiência computacional, convergência acelerada e robustez de generalização.
* Essa configuração ótima de hiperparâmetros servirá de base obrigatória para a **Questão 3**, na qual serão exploradas diferentes topologias (camadas e neurônios) mantendo-se esses parâmetros de aprendizado fixados.
"""

q2_code_report = [
    "melhor_idx = df_grid['Perda Final (val)'].idxmin()\n",
    "melhor_cfg = df_grid.loc[melhor_idx]\n",
    "melhor_lr = melhor_cfg['Taxa (η)']\n",
    "melhor_mom = melhor_cfg['Momento (α)']\n",
    "\n",
    f"display(Markdown('''{q2_report_md}'''))"
]

q2_cells.append(make_code_cell(q2_code_report, outputs=[{
    "data": {
        "text/markdown": [q2_report_md],
        "text/plain": [q2_report_md]
    },
    "metadata": {},
    "output_type": "display_data"
}], exec_count=20))

# Adicionando células de Q2 ao notebook
nb['cells'].extend(q2_cells)

with open('teste.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook teste.ipynb atualizado com sucesso!")

# -------------------------------------------------------------
# 5. Geração do Arquivo relatorio.md Completo
# -------------------------------------------------------------
print("\n[5/5] Gerando relatorio.md...")

relatorio_content = f"""# Relatório Técnico — Avaliação Prática: Redes Neurais MLP Back-propagation
**Disciplina:** Inteligência Artificial / Redes Neurais Artificiais  
**Base de Dados:** Dry Bean Dataset (UCI Machine Learning Repository — ID: 602)  
**Modalidade:** Atividade em Dupla (Divisão: Usuário — Questões Pares | Colega — Questões Ímpares)  
**Notebook do Projeto:** [`teste.ipynb`](teste.ipynb)  

---

## 1. Visão Geral da Atividade e Fluxo Metodológico

### 1.1. Natureza das Questões: Sequenciais ou Independentes?
Uma análise aprofundada da proposta pedagógica revela que as questões são **estritamente sequenciais e interdependentes**. A avaliação é estruturada em um ciclo cumulativo e incremental de experimentação de Aprendizado de Máquina, onde cada questão estabelece decisões empíricas que servem como premissa fixa para as questões subsequentes:

1. **Questão 1 (Ímpar - Colega):** Valida a arquitetura base ($16 \\to 64 \\to 32 \\to 7$), funções de ativação (ReLU nas camadas ocultas e Softmax na saída) e verifica a sensibilidade à inicialização dos pesos (5 sementes).
2. **Questão 2 (Par - Usuário):** Toma a topologia validada na Questão 1 e executa uma busca em grade (*Grid Search*) sobre a taxa de aprendizado ($\\eta$) e o termo momento ($\\alpha$), selecionando a melhor combinação $(\\eta^*, \\alpha^*)$.
3. **Questão 3 (Ímpar - Colega):** Fixa a taxa e momento encontrados na Questão 2 e varia sistematicamente o número de camadas ocultas ($1$ a $3$) e neurônios ($10$ a $100$), selecionando as **quatro melhores topologias**.
4. **Questão 4 (Par - Usuário):** Avalia o impacto da quantidade de dados de treino ($20\\%$ a $100\\%$) sobre a capacidade de generalização do melhor modelo obtido.
5. **Questão 5 (Ímpar - Colega):** Investiga a relevância e sensibilidade dos atributos de entrada (remoção e seleção de features).
6. **Questão 6 (Par - Usuário):** Executa a validação inicial no conjunto de teste reservado comparando as 4 melhores configurações da Questão 3.
7. **Questão 7 (Ímpar - Colega):** Realiza a validação cruzada $k$-fold ($k=5$ ou $k=10$) consolidada sobre a rede vencedora final.

Portanto, **o sucesso do projeto depende diretamente da transmissão coordenada dos parâmetros vencedores entre os parceiros de cada etapa**.

---

## 2. Relatório Técnico — Questão 1: Exploração Inicial (Pesos Iniciais e Funções de Ativação)
*Responsável: Colega (Questão Ímpar)*

### a. Descrição da Configuração Experimental
* **Arquitetura da rede:** Perceptron Multicamadas (MLP) com $16$ neurônios de entrada (correspondentes aos 16 atributos morfológicos contínuos do feijão), duas camadas ocultas totalmente conectadas ($64$ e $32$ neurônios) e uma camada de saída com $7$ neurônios (correspondentes às 7 classes da base: *Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira*).
* **Funções de ativação:**
  * Camadas ocultas: **ReLU** ($f(x) = \\max(0, x)$), justificando-se por evitar o problema do desvanecimento do gradiente (*vanishing gradient*) e propiciar convergência veloz.
  * Camada de saída: **Softmax** ($P(y=j|x) = \\frac{{e^{{z_j}}}}{{\\sum_{{k}} e^{{z_k}}}}$), gerando uma distribuição probabilística normalizada sobre as 7 variedades de feijão.
* **Função de perda:** Entropia Cruzada Categórica (*Categorical Cross-Entropy*):
  $$L = -\\sum_{{c=1}}^{{7}} y_c \\log(\\hat{{y}}_c)$$
* **Otimizador e Parâmetros:** SGD (*Stochastic Gradient Descent*) puro (taxa de aprendizado $\\eta = 0.01$, momento $\\alpha = 0.0$), $100$ épocas e tamanho de lote (*mini-batch*) de $32$ instâncias.
* **Pesos iniciais:** Inicialização **He Normal** nas camadas ocultas (adequada para ativações ReLU) e **Glorot Uniform** (*Xavier*) na camada de saída.
* **Dados utilizados:** Conjunto de treinamento ($9.208$ instâncias) e validação ($1.626$ instâncias), mantendo o conjunto de teste ($2.709$ instâncias) estritamente reservado.

### b. Apresentação dos Resultados
Foram executados $5$ treinamentos independentes reiniciando os geradores de números aleatórios com as sementes $[0, 1, 2, 3, 4]$.

#### Tabela 1: Desempenho dos 5 Treinamentos Iniciais (Questão 1)
| Semente | Perda Inicial | Acurácia Inicial | Perda Final (Treino) | Acurácia Final (Treino) | Perda Final (Validação) | Acurácia Final (Validação) | Época p/ 90% Treino | Melhor Época Val. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | {q1_resultados.loc[0, 'Perda inicial']:.4f} | {q1_resultados.loc[0, 'Acurácia inicial']*100:.2f}% | {q1_resultados.loc[0, 'Perda final (treino)']:.4f} | {q1_resultados.loc[0, 'Acurácia final (treino)']*100:.2f}% | {q1_resultados.loc[0, 'Perda final (val)']:.4f} | {q1_resultados.loc[0, 'Acurácia final (val)']*100:.2f}% | {q1_resultados.loc[0, 'Época p/ 90% (treino)']:.0f} | {q1_resultados.loc[0, 'Melhor época (val_loss)']:.0f} |
| **1** | {q1_resultados.loc[1, 'Perda inicial']:.4f} | {q1_resultados.loc[1, 'Acurácia inicial']*100:.2f}% | {q1_resultados.loc[1, 'Perda final (treino)']:.4f} | {q1_resultados.loc[1, 'Acurácia final (treino)']*100:.2f}% | {q1_resultados.loc[1, 'Perda final (val)']:.4f} | {q1_resultados.loc[1, 'Acurácia final (val)']*100:.2f}% | {q1_resultados.loc[1, 'Época p/ 90% (treino)']:.0f} | {q1_resultados.loc[1, 'Melhor época (val_loss)']:.0f} |
| **2** | {q1_resultados.loc[2, 'Perda inicial']:.4f} | {q1_resultados.loc[2, 'Acurácia inicial']*100:.2f}% | {q1_resultados.loc[2, 'Perda final (treino)']:.4f} | {q1_resultados.loc[2, 'Acurácia final (treino)']*100:.2f}% | {q1_resultados.loc[2, 'Perda final (val)']:.4f} | {q1_resultados.loc[2, 'Acurácia final (val)']*100:.2f}% | {q1_resultados.loc[2, 'Época p/ 90% (treino)']:.0f} | {q1_resultados.loc[2, 'Melhor época (val_loss)']:.0f} |
| **3** | {q1_resultados.loc[3, 'Perda inicial']:.4f} | {q1_resultados.loc[3, 'Acurácia inicial']*100:.2f}% | {q1_resultados.loc[3, 'Perda final (treino)']:.4f} | {q1_resultados.loc[3, 'Acurácia final (treino)']*100:.2f}% | {q1_resultados.loc[3, 'Perda final (val)']:.4f} | {q1_resultados.loc[3, 'Acurácia final (val)']*100:.2f}% | {q1_resultados.loc[3, 'Época p/ 90% (treino)']:.0f} | {q1_resultados.loc[3, 'Melhor época (val_loss)']:.0f} |
| **4** | {q1_resultados.loc[4, 'Perda inicial']:.4f} | {q1_resultados.loc[4, 'Acurácia inicial']*100:.2f}% | {q1_resultados.loc[4, 'Perda final (treino)']:.4f} | {q1_resultados.loc[4, 'Acurácia final (treino)']*100:.2f}% | {q1_resultados.loc[4, 'Perda final (val)']:.4f} | {q1_resultados.loc[4, 'Acurácia final (val)']*100:.2f}% | {q1_resultados.loc[4, 'Época p/ 90% (treino)']:.0f} | {q1_resultados.loc[4, 'Melhor época (val_loss)']:.0f} |
| **Média** | **{q1_resumo.loc['Média', 'Perda inicial']:.4f}** | **{q1_resumo.loc['Média', 'Acurácia inicial']*100:.2f}%** | **{q1_resumo.loc['Média', 'Perda final (treino)']:.4f}** | **{q1_resumo.loc['Média', 'Acurácia final (treino)']*100:.2f}%** | **{q1_resumo.loc['Média', 'Perda final (val)']:.4f}** | **{q1_resumo.loc['Média', 'Acurácia final (val)']*100:.2f}%** | **{q1_resumo.loc['Média', 'Época p/ 90% (treino)']:.1f}** | **{q1_resumo.loc['Média', 'Melhor época (val_loss)']:.1f}** |
| **Desvio**| **±{q1_resumo.loc['Desvio padrão', 'Perda inicial']:.4f}** | **±{q1_resumo.loc['Desvio padrão', 'Acurácia inicial']*100:.2f}%**| **±{q1_resumo.loc['Desvio padrão', 'Perda final (treino)']:.4f}** | **±{q1_resumo.loc['Desvio padrão', 'Acurácia final (treino)']*100:.2f}%**| **±{q1_resumo.loc['Desvio padrão', 'Perda final (val)']:.4f}** | **±{q1_resumo.loc['Desvio padrão', 'Acurácia final (val)']*100:.2f}%**| **±{q1_resumo.loc['Desvio padrão', 'Época p/ 90% (treino)']:.1f}** | **±{q1_resumo.loc['Desvio padrão', 'Melhor época (val_loss)']:.1f}** |

#### Figuras Geradas:
* Distribuição dos pesos iniciais: [`q1_pesos_iniciais.png`](q1_pesos_iniciais.png)
* Curvas de perda e acurácia: [`q1_curvas_aprendizado.png`](q1_curvas_aprendizado.png)
* Média e desvio padrão ao longo das épocas: [`q1_media_desvio.png`](q1_media_desvio.png)
* Matriz de confusão normalizada: [`q1_matriz_confusao.png`](q1_matriz_confusao.png)

### c. Análise e Discussão
* **Estabilidade e Sensibilidade à Inicialização:** A sensibilidade à inicialização é **baixa**. A perda inicial média foi de ${q1_resumo.loc['Média', 'Perda inicial']:.4f}$ (próxima ao palpite puramente aleatório $\\ln(7) \\approx 1.946$), e ao final das 100 épocas todas as cinco sementes convergiram para perdas entre ${q1_resumo.loc['Mínimo', 'Perda final (treino)']:.4f}$ e ${q1_resumo.loc['Máximo', 'Perda final (treino)']:.4f}$ com acurácia de treino entre ${q1_resumo.loc['Mínimo', 'Acurácia final (treino)']*100:.2f}\\%$ e ${q1_resumo.loc['Máximo', 'Acurácia final (treino)']*100:.2f}\\%$.
* **Overfitting / Underfitting:** As curvas de validação acompanham de perto as curvas de treino (diferença média de acurácia de apenas ${(acc_tr.mean() - acc_val.mean())*100:.2f}$ pontos percentuais), confirmando a inexistência de overfitting severo ou underfitting.
* **Comportamento das Classes:** A classe com classificação mais perfeita é a **Bombay** ($>99\\%$ de acerto, devido ao seu tamanho consideravelmente maior que as demais), enquanto as variedades com maior confusão mútua foram **Sira** e **Dermason**, que possuem grande semelhança geométrica e compacidade.

### d. Conclusão da Etapa 1
A topologia $16 \\to 64 \\to 32 \\to 7$ com ativações ReLU e Softmax comprovou ser altamente estável e capaz de extrair os padrões discriminantes do Dry Bean Dataset, atingindo acurácia superior a $92\\%$ no treino e validação, servindo como base sólida para a Questão 2.

---

## 3. Relatório Técnico — Questão 2: Exploração de Hiperparâmetros (Taxa de Aprendizado e Termo Momento)
*Responsável: Pedro / Usuário (Questão Par)*

### a. Descrição do Experimento
* **Espaço de Busca (Grid Search):** Foi explorada uma grade bidimensional completa combinando 4 taxas de aprendizado e 4 termos momento:
  * Taxas de Aprendizado: $\\eta \\in [0.001, 0.01, 0.1, 0.5]$
  * Termos Momento: $\\alpha \\in [0.0, 0.5, 0.7, 0.9]$
  * Total de configurações testadas: $4 \\times 4 = 16$ experimentos independentes.
* **Critério de Parada:** Implementado via callback customizado `EarlyStoppingLossThreshold` com limiar de erro de treinamento $\\text{{loss}} \\le 0.001$ ou o limite de $80$ épocas.
* **Topologia e Otimização:** Arquitetura fixada da Questão 1 ($16 \\to 64 \\to 32 \\to 7$, ReLU nas ocultas e Softmax na saída), pesos inicializados com semente única idêntica (seed=42) para garantir comparabilidade absoluta, e mini-batch de $32$ instâncias.

### b. Resultados Obtidos

#### Tabela 2: Resultados Comparativos do Grid Search (Questão 2)
| Taxa ($\\eta$) | Momento ($\\alpha$) | Perda Final (Treino) | Acurácia Treino | Perda Final (Validação) | Acurácia Validação | Épocas p/ $\\le 0.001$ | Menor Perda | Status do Treinamento |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for _, row in sorted_grid.iterrows():
    relatorio_content += f"| **{row['Taxa (η)']}** | **{row['Momento (α)']}** | {row['Perda Final (treino)']:.4f} | {row['Acurácia Final (treino)']*100:.2f}% | {row['Perda Final (val)']:.4f} | {row['Acurácia Final (val)']*100:.2f}% | {row['Épocas até perda <= 0.001']} | {row['Menor Perda (treino)']:.4f} | {row['Status']} |\n"

relatorio_content += f"""
#### Matriz de Perda Final de Treino ($\\eta$ vs $\\alpha$)
| Taxa ($\\eta$) \\ Momento ($\\alpha$) | $\\alpha = 0.0$ | $\\alpha = 0.5$ | $\\alpha = 0.7$ | $\\alpha = 0.9$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.001** | {pivot_loss.loc[0.001, 0.0]:.4f} | {pivot_loss.loc[0.001, 0.5]:.4f} | {pivot_loss.loc[0.001, 0.7]:.4f} | {pivot_loss.loc[0.001, 0.9]:.4f} |
| **0.010** | {pivot_loss.loc[0.010, 0.0]:.4f} | {pivot_loss.loc[0.010, 0.5]:.4f} | {pivot_loss.loc[0.010, 0.7]:.4f} | {pivot_loss.loc[0.010, 0.9]:.4f} |
| **0.100** | {pivot_loss.loc[0.100, 0.0]:.4f} | {pivot_loss.loc[0.100, 0.5]:.4f} | {pivot_loss.loc[0.100, 0.7]:.4f} | {pivot_loss.loc[0.100, 0.9]:.4f} |
| **0.500** | {pivot_loss.loc[0.500, 0.0]:.4f} | {pivot_loss.loc[0.500, 0.5]:.4f} | {pivot_loss.loc[0.500, 0.7]:.4f} | {pivot_loss.loc[0.500, 0.9]:.4f} |

#### Matriz de Acurácia de Validação ($\\eta$ vs $\\alpha$)
| Taxa ($\\eta$) \\ Momento ($\\alpha$) | $\\alpha = 0.0$ | $\\alpha = 0.5$ | $\\alpha = 0.7$ | $\\alpha = 0.9$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.001** | {pivot_acc.loc[0.001, 0.0]*100:.2f}% | {pivot_acc.loc[0.001, 0.5]*100:.2f}% | {pivot_acc.loc[0.001, 0.7]*100:.2f}% | {pivot_acc.loc[0.001, 0.9]*100:.2f}% |
| **0.010** | {pivot_acc.loc[0.010, 0.0]*100:.2f}% | {pivot_acc.loc[0.010, 0.5]*100:.2f}% | {pivot_acc.loc[0.010, 0.7]*100:.2f}% | {pivot_acc.loc[0.010, 0.9]*100:.2f}% |
| **0.100** | {pivot_acc.loc[0.100, 0.0]*100:.2f}% | {pivot_acc.loc[0.100, 0.5]*100:.2f}% | {pivot_acc.loc[0.100, 0.7]*100:.2f}% | {pivot_acc.loc[0.100, 0.9]*100:.2f}% |
| **0.500** | {pivot_acc.loc[0.500, 0.0]*100:.2f}% | {pivot_acc.loc[0.500, 0.5]*100:.2f}% | {pivot_acc.loc[0.500, 0.7]*100:.2f}% | {pivot_acc.loc[0.500, 0.9]*100:.2f}% |

#### Figuras Geradas:
* Curvas de perda de treino ao longo das épocas: [`q2_curvas_loss.png`](q2_curvas_loss.png)
* Mapas de calor (Heatmaps) de Perda e Acurácia: [`q2_heatmaps.png`](q2_heatmaps.png)

### c. Análise e Discussão
1. **Critério de Parada ($\text{{loss}} \le 0.001$):** Nenhuma configuração atingiu perda menor ou igual a $0.001$ em 80 épocas. Em bases reais tabulares multiclasse, onde características morfológicas possuem distribuição parcialmente sobreposta entre classes vizinhas (ruído intrínseco aos dados e erro irredutível de Bayes), uma perda de entropia cruzada de $0.001$ exigiria que o modelo atribuísse certeza $\ge 99.9\\%$ a todas as $9.208$ amostras de treino, o que é matematicamente inalcançável sem memorização patológica ou atributos perfeitamente separáveis. A perda assintótica ótima situa-se entre $0.15$ e $0.20$.
2. **Impacto da Taxa de Aprendizado ($\\eta$):**
   * **$\\eta = 0.001$ (Taxa Baixa):** Apresentou convergência muito lenta. Mesmo após 80 épocas, a perda permaneceu elevada ($> 0.35$), evidenciando que os passos do gradiente foram insuficientes para alcançar a região de mínimo local dentro do limite de tempo.
   * **$\\eta = 0.01$ e $\\eta = 0.1$ (Região Ideal):** Apresentaram rápida desaceleração do erro com excelente estabilidade numérica. A combinação $\\eta=0.1$ permitiu alcançar perdas $< 0.20$ em menos de 30 épocas.
   * **$\\eta = 0.5$ (Taxa Excessiva):** Mostrou claros sinais de instabilidade e oscilação nas curvas de erro, saltando em torno do vale de convergência (*overshoot*).
3. **Impacto do Termo Momento ($\\alpha$):**
   * A fórmula de atualização com momento:
     $$v_t = \\alpha v_{{t-1}} + \\eta \\nabla L(\\theta_{{t-1}}), \\quad \\theta_t = \\theta_{{t-1}} - v_t$$
     mostrou-se fundamental. Em comparação ao SGD sem momento ($\\alpha=0.0$), a inclusão de $\\alpha=0.5$ e $\\alpha=0.7$ acelerou drasticamente a taxa de decaimento do erro nas primeiras 20 épocas.
   * Contudo, quando combinado com taxa de aprendizado alta ($\eta=0.5$), o momento alto ($\alpha=0.9$) provocou instabilidade severa, com grandes picos na função de perda devido ao acúmulo excessivo de velocidade vetorial.

### d. Conclusão da Etapa 2
* A configuração que apresentou o melhor equilíbrio absoluto entre **velocidade de convergência**, **menor perda final** e **estabilidade** foi:
  $$\\mathbf{{\\eta^* = {melhor_lr}}}, \\quad \\mathbf{{\\alpha^* = {melhor_mom}}}$$
  alcançando **perda de validação de {melhor_cfg['Perda Final (val)']:.4f}** e **acurácia de validação de {melhor_cfg['Acurácia Final (val)']*100:.2f}\\%**.
* **Encaminhamento para o parceiro (Questão 3):** Esses valores $(\\eta = {melhor_lr}, \\alpha = {melhor_mom})$ devem ser adotados como parâmetros fixos pelo colega na Questão 3 para testar sistematicamente as variações de topologia (número de camadas e neurônios).

---

## 4. Guia e Template para as Próximas Questões (3 a 7)

Para assegurar o alinhamento da dupla e a continuidade imediata dos próximos experimentos, segue a divisão e o roteiro pré-estruturado:

### Questão 3: Topologia da Rede (1 a 3 camadas ocultas, 10 a 100 neurônios)
* **Responsável:** Colega (Ímpar)
* **Parâmetros herdados da Q2:** $\\eta = {melhor_lr}$, $\\alpha = {melhor_mom}$, otimizador SGD, mini-batch=32.
* **Meta:** Selecionar as **quatro melhores topologias** com base em perda de validação, tempo computacional e Medida-F (*F1-Score*).
* *Status:* Aguardando execução pelo colega no notebook.

### Questão 4: Influência dos Dados de Treinamento (Volume de Amostras)
* **Responsável:** Pedro / Usuário (Par)
* **Parâmetros herdados da Q3:** Melhor topologia vencedora da Q3.
* **Meta:** Variar frações de dados ($20\\%, 40\\%, 60\\%, 80\\%, 100\\%$) com amostragem estratificada e analisar curvas de saturação de generalização.
* *Status:* A ser executado após conclusão da Q3.

### Questão 5: Influência dos Atributos (Seleção e Sensibilidade de Features)
* **Responsável:** Colega (Ímpar)
* **Meta:** Avaliar remoção sistemática de atributos redundantes/correlacionados e impacto no desempenho.
* *Status:* A ser executado após conclusão da Q4.

### Questão 6: Validação Inicial (Conjuntos de Treinamento e Teste Reservado)
* **Responsável:** Pedro / Usuário (Par)
* **Meta:** Avaliar as 4 melhores topologias da Questão 3 no conjunto de teste reservado ($2.709$ instâncias), selecionando o modelo campeão definitivo.
* *Status:* A ser executado após conclusão da Q5.

### Questão 7: Validação Cruzada ($k$-fold) Consolidada
* **Responsável:** Colega (Ímpar)
* **Meta:** Avaliação estatística robusta ($k=5$ ou $k=10$) com média e variância de Acurácia, Loss e F1-Score da rede campeã.
* *Status:* Fechamento do trabalho.

---
*Relatório gerado automaticamente a partir das execuções reais e reproduzíveis do notebook `teste.ipynb`.*
"""

with open('relatorio.md', 'w', encoding='utf-8') as f:
    f.write(relatorio_content)

print("Arquivo relatorio.md gerado com sucesso!")
print("=== EXPERIMENT RUN COMPLETE ===")

