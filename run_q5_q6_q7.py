"""
Questões 5, 6 e 7 — Influência dos Atributos, Validação Inicial (Treino/Teste) e Validação Cruzada (k-fold).

Executa as células num kernel Jupyter real (com as células prévias do notebook teste.ipynb),
grava as saídas reais e anexa as células ao final de `teste.ipynb` com a tag 'q5q6q7'.

Uso:
    python run_q5_q6_q7.py            # executa tudo e atualiza o notebook
    python run_q5_q6_q7.py --rebuild  # apenas remonta o notebook a partir do cache
    python run_q5_q6_q7.py --smoke    # execução rápida de teste
"""
import os
import sys
import json
import shutil
import nbformat

PROJ = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(PROJ, 'teste.ipynb')
OUT_DIR = os.path.join(PROJ, 'resultados_q5q6q7')
CACHE = os.path.join(OUT_DIR, 'celulas_q5q6q7.json')
TAG = 'q5q6q7'

# ---------------------------------------------------------------------------------------------
# Código executado antes das células no kernel
# ---------------------------------------------------------------------------------------------
PRELUDE_EXTRA = r'''
tf.get_logger().setLevel('ERROR')

def reset_seeds(seed):
    keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

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
'''

CELLS = []
md = lambda s: CELLS.append({'type': 'markdown', 'src': s.strip('\n')})
code = lambda s: CELLS.append({'type': 'code', 'src': s.strip('\n')})
relato = lambda f: CELLS.append({'type': 'relato', 'file': f})

# =============================================================================================
# QUESTÃO 5: INFLUÊNCIA DOS ATRIBUTOS (QUANTIDADE, QUALIDADE E SENSIBILIDADE)
# =============================================================================================

md(r'''
# 5. Influência dos Atributos — Quantidade, Qualidade e Sensibilidade das Features

Nesta etapa, investigamos a relevância, redundância e sensibilidade dos 16 atributos morfológicos e geométricos do **Dry Bean Dataset** sobre o aprendizado da rede neural MLP.
Mantendo a arquitetura de melhor compromisso obtida na Questão 3 (**30-30**, com ativações ReLU e Softmax) e os hiperparâmetros fixados na Questão 2 ($\eta^* = 0.01$, $\alpha^* = 0.5$, lote 32), avaliamos:

1. **Multicolinearidade e Correlação de Pearson:** identificação de atributos geometricamente redundantes ($|r| > 0.95$);
2. **Ranking de Importância Estatística:** teste F de ANOVA (*one-way ANOVA*) e Informação Mútua (*Mutual Information*);
3. **Subconjuntos Estratégicos de Features:**
   * **Baseline completo (16 atributos)**;
   * **Sem Colineares ($|r| < 0.95$, 11 atributos)**: remoção de variáveis de mesma dimensão geométrica (`ConvexArea`, `EquivDiameter`, `ShapeFactor3`, `Perimeter`, `Eccentricity`);
   * **Apenas Forma (Shape Only, 10 atributos)**: fatores invariantes à escala absoluta (`AspectRatio`, `Eccentricity`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1-4`);
   * **Apenas Dimensão (Size Only, 6 atributos)**: medidas lineares e de área (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `ConvexArea`, `EquivDiameter`);
   * **Top-8 Mais Relevantes**: as 8 melhores características pelo ranking estatístico;
   * **Top-4 Essenciais**: o conjunto mínimo de alta discriminação;
   * **Sensibilidade ao Ruído**: os 16 atributos originais acrescidos de 2 variáveis aleatórias gaussianas $\mathcal{N}(0, 1)$ como ruído puro.
''')

code(r'''
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

os.makedirs('resultados_q5q6q7', exist_ok=True)

# 1. Análise de Correlação e Multicolinearidade
df_features = pd.DataFrame(X, columns=feats)
matriz_corr = df_features.corr()

# Ranking estatístico via ANOVA F-value
f_vals, p_vals = f_classif(X_tr, y_tr)
df_importancia = pd.DataFrame({
    'Atributo': feats,
    'F_Score': f_vals,
    'p_valor': p_vals
}).sort_values('F_Score', ascending=False).reset_index(drop=True)

print("=== Ranking de Atributos por ANOVA F-Score ===")
for i, r in df_importancia.iterrows():
    print(f"{i+1:2d}. {r['Atributo']:16s} | F-Score: {r['F_Score']:8.2f} (p < {r['p_valor']:.1e})")

# Pares com alta colinearidade (|r| > 0.95)
colineares = []
for i in range(len(feats)):
    for j in range(i + 1, len(feats)):
        r = matriz_corr.iloc[i, j]
        if abs(r) >= 0.95:
            colineares.append((feats[i], feats[j], r))

print(f"\nPares com alta colinearidade (|r| >= 0.95): {len(colineares)}")
for f1, f2, r in colineares:
    print(f"  - {f1:16s} <-> {f2:16s}: r = {r:+.4f}")

# Gráfico da Matriz de Correlação e Ranking
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

im = axes[0].imshow(matriz_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(feats)), feats, rotation=90, fontsize=9)
axes[0].set_yticks(range(len(feats)), feats, fontsize=9)
axes[0].set_title('Matriz de Correlação de Pearson (16 Atributos)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# Ranking
cores_rank = plt.cm.viridis(np.linspace(0.2, 0.85, len(df_importancia)))[::-1]
axes[1].barh(df_importancia['Atributo'][::-1], df_importancia['F_Score'][::-1], color=cores_rank)
axes[1].set_xlabel('Valor F de ANOVA', fontsize=11)
axes[1].set_title('Ranking de Relevância dos Atributos (ANOVA F-Score)', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(df_importancia['F_Score'][::-1]):
    axes[1].text(v + 100, i, f"{v:.0f}", va='center', fontsize=8)

plt.tight_layout()
fig.savefig('q5_correlacao_importancia.png', dpi=150)
fig.savefig('resultados_q5q6q7/q5_correlacao_importancia.png', dpi=150)
plt.show()
''')

code(r'''
# Definição sistemática dos subconjuntos de atributos
top8_feats = df_importancia['Atributo'].head(8).tolist()
top4_feats = df_importancia['Atributo'].head(4).tolist()

# Remoção de variáveis estritamente redundantes com |r| > 0.95
# ConvexArea e EquivDiameter colineares com Area; ShapeFactor3 colinear com Compactness; Perimeter colinear com MajorAxis
sem_colineares = [f for f in feats if f not in ['ConvexArea', 'EquivDiameter', 'ShapeFactor3', 'Perimeter', 'Eccentricity']]

apenas_forma = ['AspectRatio', 'Eccentricity', 'Extent', 'Solidity', 'Roundness',
                'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']
apenas_tamanho = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'ConvexArea', 'EquivDiameter']

SUBCONJUNTOS = {
    '1. Todos (16 atributos)': list(feats),
    '2. Sem Colineares (11 atrib.)': sem_colineares,
    '3. Apenas Forma (10 atrib.)': apenas_forma,
    '4. Apenas Dimensão (6 atrib.)': apenas_tamanho,
    '5. Top-8 Mais Relevantes': top8_feats,
    '6. Top-4 Essenciais': top4_feats,
    '7. 16 Atrib. + 2 Ruídos Gauss.': list(feats) + ['_ruido1', '_ruido2']
}

SEEDS_Q5 = [0, 1, 2]
MAX_EPOCHS_Q5 = 80
STOP_LOSS = 0.001
ETA, ALPHA = 0.01, 0.5

# Adicionando ruídos gaussianos no dataset para o experimento 7
np.random.seed(42)
ruido_tr = np.random.randn(len(X_tr), 2).astype('float32')
ruido_val = np.random.randn(len(X_val), 2).astype('float32')
ruido_test = np.random.randn(len(X_test), 2).astype('float32')

df_X_tr = pd.DataFrame(X_tr, columns=feats)
df_X_val = pd.DataFrame(X_val, columns=feats)
df_X_test = pd.DataFrame(X_test, columns=feats)

df_X_tr['_ruido1'], df_X_tr['_ruido2'] = ruido_tr[:, 0], ruido_tr[:, 1]
df_X_val['_ruido1'], df_X_val['_ruido2'] = ruido_val[:, 0], ruido_val[:, 1]
df_X_test['_ruido1'], df_X_test['_ruido2'] = ruido_test[:, 0], ruido_test[:, 1]

def treinar_subconjunto(cols, seed):
    reset_seeds(seed)
    
    # Padronização restrita ao subconjunto
    sc = StandardScaler().fit(df_X_tr[cols])
    X_tr_sub = sc.transform(df_X_tr[cols]).astype('float32')
    X_val_sub = sc.transform(df_X_val[cols]).astype('float32')
    X_test_sub = sc.transform(df_X_test[cols]).astype('float32')
    
    n_in = len(cols)
    model = keras.Sequential([
        keras.Input(shape=(n_in,)),
        layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.HeNormal(seed=seed), name='oculta_1'),
        layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.HeNormal(seed=seed+1), name='oculta_2'),
        layers.Dense(n_classes, activation='softmax', kernel_initializer=keras.initializers.GlorotUniform(seed=seed+100), name='saida')
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=ETA, momentum=ALPHA),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'], steps_per_execution=32)
    
    cb = EarlyStoppingLossThreshold(STOP_LOSS)
    t0 = time.perf_counter()
    h = model.fit(X_tr_sub, Y_tr, validation_data=(X_val_sub, Y_val),
                  epochs=MAX_EPOCHS_Q5, batch_size=32, callbacks=[cb], verbose=0, shuffle=True)
    t_total = time.perf_counter() - t0
    
    prob_val = model.predict(X_val_sub, verbose=0, batch_size=1024)
    pred_val = prob_val.argmax(axis=1)
    prob_test = model.predict(X_test_sub, verbose=0, batch_size=1024)
    pred_test = prob_test.argmax(axis=1)
    
    hist = pd.DataFrame(h.history); hist.index += 1
    return {
        'hist': hist,
        'loss_tr': hist['loss'].iloc[-1],
        'loss_val': hist['val_loss'].iloc[-1],
        'acc_val': accuracy_score(y_val, pred_val),
        'f1_val': f1_score(y_val, pred_val, average='macro'),
        'acc_test': accuracy_score(y_test, pred_test),
        'f1_test': f1_score(y_test, pred_test, average='macro'),
        'prec_test': precision_score(y_test, pred_test, average='macro', zero_division=0),
        'rec_test': recall_score(y_test, pred_test, average='macro', zero_division=0),
        'tempo': t_total
    }

print("Iniciando treinamento sistemático dos 7 subconjuntos de atributos...")
q5_registros = []
q5_historicos = {}

for nome_sub, cols in SUBCONJUNTOS.items():
    print(f"\n-> Subconjunto: {nome_sub} ({len(cols)} atributos)")
    for seed in SEEDS_Q5:
        res = treinar_subconjunto(cols, seed)
        q5_registros.append({
            'Subconjunto': nome_sub,
            'Qtd_Atributos': len(cols),
            'Semente': seed,
            'Perda treino': res['loss_tr'],
            'Perda val': res['loss_val'],
            'Acurácia val': res['acc_val'],
            'F1 val': res['f1_val'],
            'Acurácia teste': res['acc_test'],
            'F1 teste': res['f1_test'],
            'Precisão teste': res['prec_test'],
            'Revocação teste': res['rec_test'],
            'Tempo (s)': res['tempo']
        })
        if seed == 0:
            q5_historicos[nome_sub] = res['hist']
        print(f"   Seed {seed} | Loss Val: {res['loss_val']:.4f} | F1 Val: {res['f1_val']:.4f} | F1 Teste: {res['f1_test']:.4f} ({res['tempo']:.1f}s)")

df_q5 = pd.DataFrame(q5_registros)
df_q5.to_csv('resultados_q5q6q7/q5_atributos.csv', index=False)
''')

code(r'''
# Tabela de Resultados Consolidados da Questão 5
q5_media = df_q5.groupby(['Subconjunto', 'Qtd_Atributos'], sort=False).mean(numeric_only=True).drop(columns=['Semente'])
q5_dp = df_q5.groupby(['Subconjunto', 'Qtd_Atributos'], sort=False).std(numeric_only=True).drop(columns=['Semente'])

q5_resumo = pd.DataFrame(index=q5_media.index)
q5_resumo['Perda val'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q5_media['Perda val'], q5_dp['Perda val'])]
q5_resumo['Acurácia val'] = [f"{m*100:.2f}% ± {s*100:.2f}" for m, s in zip(q5_media['Acurácia val'], q5_dp['Acurácia val'])]
q5_resumo['F1 val'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q5_media['F1 val'], q5_dp['F1 val'])]
q5_resumo['F1 teste'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q5_media['F1 teste'], q5_dp['F1 teste'])]
q5_resumo['Tempo médio'] = [f"{m:.1f} s" for m in q5_media['Tempo (s)']]

display(Markdown('### Tabela 7 — Comparativo dos Subconjuntos de Atributos (Média ± Desvio de 3 Sementes)'))
display(q5_resumo)

# Gráfico 1: Curvas de Convergência ao Longo das Épocas
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
cores_sub = plt.cm.tab10.colors

sub_destaque = [
    '1. Todos (16 atributos)',
    '2. Sem Colineares (11 atrib.)',
    '3. Apenas Forma (10 atrib.)',
    '4. Apenas Dimensão (6 atrib.)',
    '5. Top-8 Mais Relevantes',
    '7. 16 Atrib. + 2 Ruídos Gauss.'
]

for idx, nome_s in enumerate(sub_destaque):
    if nome_s in q5_historicos:
        h = q5_historicos[nome_s]
        axes[0].plot(h['val_loss'], label=nome_s, color=cores_sub[idx], linewidth=1.8)
        axes[1].plot(h['val_accuracy'], label=nome_s, color=cores_sub[idx], linewidth=1.8)

axes[0].set_title('Perda de Validação ao Longo das Épocas', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Entropia Cruzada'); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

axes[1].set_title('Acurácia de Validação ao Longo das Épocas', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Acurácia'); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)

plt.tight_layout()
fig.savefig('q5_curvas_convergencia.png', dpi=150)
fig.savefig('resultados_q5q6q7/q5_curvas_convergencia.png', dpi=150)
plt.show()

# Gráfico 2: Desempenho e Tempo em Função dos Subconjuntos
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sub_nomes_curtos = [s.split('(')[0].strip() for s in q5_media.index.get_level_values(0)]
x_pos = np.arange(len(sub_nomes_curtos))

# F1 Val e F1 Teste
axes[0].bar(x_pos - 0.18, q5_media['F1 val'], 0.36, yerr=q5_dp['F1 val'], capsize=3, label='F1 Validação', color='tab:orange')
axes[0].bar(x_pos + 0.18, q5_media['F1 teste'], 0.36, yerr=q5_dp['F1 teste'], capsize=3, label='F1 Teste Reservado', color='tab:green')
axes[0].set_xticks(x_pos, sub_nomes_curtos, rotation=35, ha='right', fontsize=9)
axes[0].set_ylabel('Medida-F (Macro)')
axes[0].set_ylim(0.85, 0.98)
axes[0].set_title('Medida-F por Subconjunto de Atributos', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3); axes[0].legend()

# Tempo vs Qtd Atributos
qtds = q5_media.index.get_level_values(1)
axes[1].plot(qtds, q5_media['Tempo (s)'], 'o--', color='tab:purple', linewidth=2, markersize=8)
axes[1].set_xlabel('Quantidade de Atributos de Entrada')
axes[1].set_ylabel('Tempo Médio de Treino (s)')
axes[1].set_title('Custo Computacional vs Dimensionalidade', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
for q, t in zip(qtds, q5_media['Tempo (s)']):
    axes[1].text(q, t + 0.3, f"{t:.1f}s", ha='center', fontsize=9)

plt.tight_layout()
fig.savefig('q5_comparativo_subconjuntos.png', dpi=150)
fig.savefig('resultados_q5q6q7/q5_comparativo_subconjuntos.png', dpi=150)
plt.show()
''')

relato('relato_q5.md')

# =============================================================================================
# QUESTÃO 6: VALIDAÇÃO INICIAL — CONJUNTOS DE TREINAMENTO E TESTE
# =============================================================================================

md(r'''
# 6. Validação Inicial — Conjuntos de Treinamento e Teste

Nesta etapa, realizamos a **Validação Inicial** comparativa entre as **quatro melhores configurações topológicas** selecionadas na Questão 3:
1. **30-30** (2 camadas ocultas: 30 e 30 neurônios, 1.657 parâmetros);
2. **100** (1 camada oculta com 100 neurônios, 2.407 parâmetros);
3. **50** (1 camada oculta com 50 neurônios, 1.207 parâmetros);
4. **30** (1 camada oculta com 30 neurônios, 727 parâmetros).

### Procedimento Experimental
* Todas as quatro redes são treinadas no **conjunto de treinamento completo** ($9.208$ exemplos), com otimizador SGD ($\eta=0.01$, $\alpha=0.5$), tamanho de lote 32 e 100 épocas.
* Cada arquitetura é executada com 3 sementes de inicialização (`SEEDS_Q6 = [0, 1, 2]`) para mitigar o viés de ponto de partida.
* A cada época, monitoram-se a **perda e a acurácia no treinamento e no conjunto de teste reservado** ($2.709$ instâncias), revelando diretamente a trajetória de generalização e a presença de eventuais divergências ou *overfitting*.
* Ao final, seleciona-se a **rede campeã definitiva** que representará a solução na Validação Cruzada ($k$-fold).
''')

code(r'''
TOP_4_CONFIGS = {
    '30-30': [30, 30],
    '100': [100],
    '50': [50],
    '30': [30]
}

SEEDS_Q6 = [0, 1, 2]
MAX_EPOCHS_Q6 = 100

# Preparação das matrizes padronizadas do teste reservado
X_test_s = scaler.transform(X_test).astype('float32')
Y_test_cat = keras.utils.to_categorical(y_test, n_classes)

q6_registros = []
q6_curvas = {}
q6_modelos = {}

print("Iniciando Validação Inicial das 4 melhores topologias no conjunto de teste...")

for nome_top, hu in TOP_4_CONFIGS.items():
    print(f"\n-> Avaliando Topologia: {nome_top} ({hu})")
    curvas_top = []
    
    for seed in SEEDS_Q6:
        reset_seeds(seed)
        model = keras.Sequential([keras.Input(shape=(n_features,))])
        for i, units in enumerate(hu):
            model.add(layers.Dense(units, activation='relu',
                                   kernel_initializer=keras.initializers.HeNormal(seed=seed + i),
                                   name=f'oculta_{i+1}'))
        model.add(layers.Dense(n_classes, activation='softmax',
                               kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 100),
                               name='saida'))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'], steps_per_execution=32)
        
        # Treinamento com acompanhamento direto do teste
        # Note: Y_test_cat para validação direta
        Y_test_cat = keras.utils.to_categorical(y_test, n_classes)
        t0 = time.perf_counter()
        h = model.fit(X_tr_s, Y_tr, validation_data=(X_test_s, Y_test_cat),
                      epochs=MAX_EPOCHS_Q6, batch_size=32, verbose=0, shuffle=True)
        t_total = time.perf_counter() - t0
        
        hist = pd.DataFrame(h.history); hist.index += 1
        curvas_top.append(hist)
        
        prob_tr = model.predict(X_tr_s, verbose=0, batch_size=1024)
        pred_tr = prob_tr.argmax(axis=1)
        prob_test = model.predict(X_test_s, verbose=0, batch_size=1024)
        pred_test = prob_test.argmax(axis=1)
        
        acc_tr = accuracy_score(y_tr, pred_tr)
        loss_tr = hist['loss'].iloc[-1]
        acc_te = accuracy_score(y_test, pred_test)
        loss_te = hist['val_loss'].iloc[-1]
        f1_te = f1_score(y_test, pred_test, average='macro')
        prec_te = precision_score(y_test, pred_test, average='macro', zero_division=0)
        rec_te = recall_score(y_test, pred_test, average='macro', zero_division=0)
        
        gap_acc = (acc_tr - acc_te) * 100
        gap_loss = loss_te - loss_tr
        
        q6_registros.append({
            'Topologia': nome_top,
            'Camadas': len(hu),
            'Parâmetros': model.count_params(),
            'Semente': seed,
            'Perda treino': loss_tr,
            'Acurácia treino': acc_tr,
            'Perda teste': loss_te,
            'Acurácia teste': acc_te,
            'F1 teste': f1_te,
            'Precisão teste': prec_te,
            'Revocação teste': rec_te,
            'Gap Acurácia (p.p.)': gap_acc,
            'Gap Perda': gap_loss,
            'Tempo (s)': t_total
        })
        if seed == 0:
            q6_modelos[nome_top] = (model, pred_test)
            
        print(f"   Seed {seed} | Loss Teste: {loss_te:.4f} | Acc Teste: {acc_te*100:.2f}% | F1 Teste: {f1_te:.4f} | Gap Acc: {gap_acc:+.2f} p.p.")
        
    q6_curvas[nome_top] = curvas_top

df_q6 = pd.DataFrame(q6_registros)
df_q6.to_csv('resultados_q5q6q7/q6_validacao_inicial.csv', index=False)
''')

code(r'''
# Consolidação e Comparação das 4 Topologias
q6_media = df_q6.groupby('Topologia', sort=False).mean(numeric_only=True).drop(columns=['Semente'])
q6_dp = df_q6.groupby('Topologia', sort=False).std(numeric_only=True).drop(columns=['Semente'])

q6_resumo = pd.DataFrame(index=q6_media.index)
q6_resumo['Parâmetros'] = q6_media['Parâmetros'].astype(int)
q6_resumo['Perda treino'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q6_media['Perda treino'], q6_dp['Perda treino'])]
q6_resumo['Perda teste'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q6_media['Perda teste'], q6_dp['Perda teste'])]
q6_resumo['Acurácia teste'] = [f"{m*100:.2f}% ± {s*100:.2f}" for m, s in zip(q6_media['Acurácia teste'], q6_dp['Acurácia teste'])]
q6_resumo['F1 teste'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q6_media['F1 teste'], q6_dp['F1 teste'])]
q6_resumo['Precisão teste'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q6_media['Precisão teste'], q6_dp['Precisão teste'])]
q6_resumo['Revocação teste'] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(q6_media['Revocação teste'], q6_dp['Revocação teste'])]
q6_resumo['Gap Acurácia'] = [f"{m:+.2f} p.p." for m in q6_media['Gap Acurácia (p.p.)']]
q6_resumo['Tempo (s)'] = [f"{m:.1f} s" for m in q6_media['Tempo (s)']]

display(Markdown('### Tabela 8 — Comparativo das Quatro Topologias no Conjunto de Teste (Média ± Desvio)'))
display(q6_resumo)

# Gráfico 1: Curvas de Validação (Treino vs Teste) ao longo das épocas
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes_flat = axes.ravel()

for i, (nome_top, curvas) in enumerate(q6_curvas.items()):
    ax = axes_flat[i]
    # Média das 3 sementes
    loss_tr_m = pd.concat([c['loss'] for c in curvas], axis=1).mean(axis=1)
    loss_te_m = pd.concat([c['val_loss'] for c in curvas], axis=1).mean(axis=1)
    acc_tr_m = pd.concat([c['accuracy'] for c in curvas], axis=1).mean(axis=1)
    acc_te_m = pd.concat([c['val_accuracy'] for c in curvas], axis=1).mean(axis=1)
    
    ax.plot(loss_tr_m, label='Perda Treino', color='tab:blue', linewidth=1.8)
    ax.plot(loss_te_m, label='Perda Teste', color='tab:orange', linewidth=1.8, linestyle='--')
    ax.set_title(f"Topologia {nome_top} ({q6_media.loc[nome_top, 'Parâmetros']:.0f} parâmetros)", fontweight='bold')
    ax.set_xlabel('Época')
    ax.set_ylabel('Entropia Cruzada Categórica')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

plt.suptitle('Curvas de Aprendizado: Perda de Treinamento vs Teste ao Longo das Épocas', fontsize=14, y=0.99)
plt.tight_layout()
fig.savefig('q6_curvas_treino_teste.png', dpi=150)
fig.savefig('resultados_q5q6q7/q6_curvas_treino_teste.png', dpi=150)
plt.show()

# Gráfico 2: Comparativo de Generalização e Matriz de Confusão
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
tops = list(TOP_4_CONFIGS.keys())
x_ind = np.arange(len(tops))

axes[0].bar(x_ind - 0.18, q6_media['Acurácia teste'] * 100, 0.36, yerr=q6_dp['Acurácia teste']*100,
            capsize=4, label='Acurácia Teste (%)', color='tab:blue')
axes[0].bar(x_ind + 0.18, q6_media['F1 teste'] * 100, 0.36, yerr=q6_dp['F1 teste']*100,
            capsize=4, label='Medida-F Teste (×100)', color='tab:green')
axes[0].set_xticks(x_ind, tops, fontsize=11)
axes[0].set_ylabel('Desempenho (%)')
axes[0].set_ylim(90, 96)
axes[0].set_title('Desempenho Global no Conjunto de Teste', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3); axes[0].legend()

# Matriz de Confusão no Teste da Topologia Vencedora
campea_nome = q6_media['F1 teste'].idxmax()
_, campea_preds = q6_modelos[campea_nome]
cm_te = confusion_matrix(y_test, campea_preds, normalize='true')
nomes_classes = [classes[k] for k in range(n_classes)]

im_cm = axes[1].imshow(cm_te, cmap='Blues', vmin=0, vmax=1)
axes[1].set_xticks(range(n_classes), nomes_classes, rotation=45, ha='right', fontsize=9)
axes[1].set_yticks(range(n_classes), nomes_classes, fontsize=9)
axes[1].set_title(f'Matriz de Confusão no Teste — Rede Campeã ({campea_nome})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Classe Prevista'); axes[1].set_ylabel('Classe Real')
for r_i in range(n_classes):
    for c_j in range(n_classes):
        axes[1].text(c_j, r_i, f"{cm_te[r_i, c_j]:.2f}", ha='center', va='center',
                     color='white' if cm_te[r_i, c_j] > 0.5 else 'black', fontsize=8)
plt.colorbar(im_cm, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig('q6_comparativo_top4.png', dpi=150)
fig.savefig('resultados_q5q6q7/q6_comparativo_top4.png', dpi=150)
fig.savefig('q6_matriz_confusao_teste.png', dpi=150)
fig.savefig('resultados_q5q6q7/q6_matriz_confusao_teste.png', dpi=150)
plt.show()

print(f"\n>>> REDE CAMPEÃ ELEITA NA QUESTÃO 6: {campea_nome}")
print(f"    F1 Teste: {q6_media.loc[campea_nome, 'F1 teste']:.4f} | Acurácia Teste: {q6_media.loc[campea_nome, 'Acurácia teste']*100:.2f}% | Parâmetros: {q6_media.loc[campea_nome, 'Parâmetros']:.0f}")
''')

relato('relato_q6.md')

# =============================================================================================
# QUESTÃO 7: VALIDAÇÃO CRUZADA (K-FOLD) CONSOLIDADA
# =============================================================================================

md(r'''
# 7. Validação Cruzada e Comparativa de Configurações Finais — Avaliação Consolidada

Para consolidar a avaliação final da solução, realizamos a **Validação Cruzada Estratificada $k$-fold com $k = 10$ partições** sobre a rede campeã eleita na Questão 6.

### Metodologia de Validação Cruzada:
* **Particionamento:** Os dados são divididos em $10$ partições disjuntas e estratificadas (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`), garantindo que a proporção das 7 classes seja rigorosamente idêntica em cada partição.
* **Prevenção de Vazamento de Dados (*Data Leakage*):** Em cada iteração $k$, o objeto `StandardScaler` é ajustado **exclusivamente nas $9$ partições de treinamento** ($\approx 12.250$ instâncias) e em seguida aplicado para transformar a partição de teste ($\approx 1.361$ instâncias).
* **Treinamento:** Cada fold executa 80 épocas com a taxa e momento validados ($\eta=0.01$, $\alpha=0.5$, lote 32), registrando a trajetória completa da perda e da acurácia.
* **Métricas Consolidadas:** Ao final das 10 execuções, calculam-se a média ($\mu$), o desvio padrão ($\sigma$), a variância ($\sigma^2$), o coeficiente de variação ($CV$) e o intervalo de confiança a $95\%$ para todas as métricas essenciais (Perda, Acurácia, Medida-F, Precisão e Revocação).
* **Matriz de Confusão Acumulada:** Todas as predições *out-of-fold* são consolidadas, permitindo avaliar o comportamento preditivo da rede sobre as $13.611$ instâncias da base completa.
''')

code(r'''
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats

K_FOLDS = 10
EPOCHS_Q7 = 80
Q7_SEED = 42

# Recuperando a configuração da rede campeã da Questão 6
campea_units = TOP_4_CONFIGS[campea_nome]
print(f"Executando Validação Cruzada 10-Fold para a Rede Campeã: {campea_nome} ({campea_units})...")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=Q7_SEED)

q7_registros = []
q7_historicos_folds = []
oof_predicoes = np.zeros(len(X), dtype=int)
oof_probabilidades = np.zeros((len(X), n_classes), dtype='float32')

for fold_idx, (idx_tr, idx_te) in enumerate(skf.split(X, Y)):
    fold_num = fold_idx + 1
    t0_fold = time.perf_counter()
    
    # Divisão e normalização estrita
    X_f_tr, y_f_tr = X[idx_tr], Y[idx_tr]
    X_f_te, y_f_te = X[idx_te], Y[idx_te]
    
    scaler_fold = StandardScaler().fit(X_f_tr)
    X_f_tr_s = scaler_fold.transform(X_f_tr).astype('float32')
    X_f_te_s = scaler_fold.transform(X_f_te).astype('float32')
    
    Y_f_tr_cat = keras.utils.to_categorical(y_f_tr, n_classes)
    Y_f_te_cat = keras.utils.to_categorical(y_f_te, n_classes)
    
    # Construção da rede com semente específica para o fold
    reset_seeds(Q7_SEED + fold_num)
    model_fold = keras.Sequential([keras.Input(shape=(n_features,))])
    for i, units in enumerate(campea_units):
        model_fold.add(layers.Dense(units, activation='relu',
                                    kernel_initializer=keras.initializers.HeNormal(seed=Q7_SEED + fold_num + i),
                                    name=f'oculta_{i+1}'))
    model_fold.add(layers.Dense(n_classes, activation='softmax',
                                kernel_initializer=keras.initializers.GlorotUniform(seed=Q7_SEED + fold_num + 100),
                                name='saida'))
    model_fold.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
                       loss=keras.losses.CategoricalCrossentropy(),
                       metrics=['accuracy'], steps_per_execution=32)
    
    cb_fold = EarlyStoppingLossThreshold(STOP_LOSS)
    h = model_fold.fit(X_f_tr_s, Y_f_tr_cat, validation_data=(X_f_te_s, Y_f_te_cat),
                       epochs=EPOCHS_Q7, batch_size=32, callbacks=[cb_fold], verbose=0, shuffle=True)
    t_fold = time.perf_counter() - t0_fold
    
    hist_f = pd.DataFrame(h.history); hist_f.index += 1
    q7_historicos_folds.append(hist_f)
    
    # Predições no fold de teste
    prob_fold = model_fold.predict(X_f_te_s, verbose=0, batch_size=1024)
    pred_fold = prob_fold.argmax(axis=1)
    
    oof_probabilidades[idx_te] = prob_fold
    oof_predicoes[idx_te] = pred_fold
    
    loss_te = log_loss(y_f_te, prob_fold, labels=range(n_classes))
    acc_te = accuracy_score(y_f_te, pred_fold)
    f1_te = f1_score(y_f_te, pred_fold, average='macro')
    prec_te = precision_score(y_f_te, pred_fold, average='macro', zero_division=0)
    rec_te = recall_score(y_f_te, pred_fold, average='macro', zero_division=0)
    
    q7_registros.append({
        'Fold': f"Fold {fold_num:02d}",
        'Exemplos Treino': len(idx_tr),
        'Exemplos Teste': len(idx_te),
        'Perda Treino': hist_f['loss'].iloc[-1],
        'Perda Teste': loss_te,
        'Acurácia Teste': acc_te,
        'F1 Teste': f1_te,
        'Precisão Teste': prec_te,
        'Revocação Teste': rec_te,
        'Tempo (s)': t_fold
    })
    print(f"  Fold {fold_num:02d} | Perda Teste: {loss_te:.4f} | Acc Teste: {acc_te*100:.2f}% | F1 Teste: {f1_te:.4f} | Tempo: {t_fold:.1f}s")

df_q7 = pd.DataFrame(q7_registros)
df_q7.to_csv('resultados_q5q6q7/q7_cv_folds.csv', index=False)
''')

code(r'''
# Estatísticas Consolidadas da Validação Cruzada
numeric_cols = ['Perda Treino', 'Perda Teste', 'Acurácia Teste', 'F1 Teste', 'Precisão Teste', 'Revocação Teste', 'Tempo (s)']
medias = df_q7[numeric_cols].mean()
desvios = df_q7[numeric_cols].std()
variancias = df_q7[numeric_cols].var()
cv_percent = (desvios / medias) * 100

# Intervalos de confiança (t-Student, 95%, gl = 9)
t_crit = stats.t.ppf(0.975, df=K_FOLDS - 1)
ic_95 = t_crit * (desvios / np.sqrt(K_FOLDS))

# Tabela consolidada com formatação
df_q7_apresentacao = df_q7.copy()
df_q7_apresentacao['Acurácia Teste'] = df_q7_apresentacao['Acurácia Teste'].map(lambda v: f"{v*100:.2f}%")
for col in ['Perda Treino', 'Perda Teste', 'F1 Teste', 'Precisão Teste', 'Revocação Teste']:
    df_q7_apresentacao[col] = df_q7_apresentacao[col].map(lambda v: f"{v:.4f}")
df_q7_apresentacao['Tempo (s)'] = df_q7_apresentacao['Tempo (s)'].map(lambda v: f"{v:.1f} s")

# Linhas de síntese
linha_media = {
    'Fold': '**Média (μ)**', 'Exemplos Treino': f"{df_q7['Exemplos Treino'].mean():.0f}", 'Exemplos Teste': f"{df_q7['Exemplos Teste'].mean():.0f}",
    'Perda Treino': f"{medias['Perda Treino']:.4f}", 'Perda Teste': f"{medias['Perda Teste']:.4f}",
    'Acurácia Teste': f"{medias['Acurácia Teste']*100:.2f}%", 'F1 Teste': f"{medias['F1 Teste']:.4f}",
    'Precisão Teste': f"{medias['Precisão Teste']:.4f}", 'Revocação Teste': f"{medias['Revocação Teste']:.4f}",
    'Tempo (s)': f"{medias['Tempo (s)']:.1f} s"
}
linha_dp = {
    'Fold': '**Desvio Padrão (σ)**', 'Exemplos Treino': '-', 'Exemplos Teste': '-',
    'Perda Treino': f"±{desvios['Perda Treino']:.4f}", 'Perda Teste': f"±{desvios['Perda Teste']:.4f}",
    'Acurácia Teste': f"±{desvios['Acurácia Teste']*100:.2f}%", 'F1 Teste': f"±{desvios['F1 Teste']:.4f}",
    'Precisão Teste': f"±{desvios['Precisão Teste']:.4f}", 'Revocação Teste': f"±{desvios['Revocação Teste']:.4f}",
    'Tempo (s)': f"±{desvios['Tempo (s)']:.1f} s"
}
linha_var = {
    'Fold': '**Variância (σ²)**', 'Exemplos Treino': '-', 'Exemplos Teste': '-',
    'Perda Treino': f"{variancias['Perda Treino']:.2e}", 'Perda Teste': f"{variancias['Perda Teste']:.2e}",
    'Acurácia Teste': f"{variancias['Acurácia Teste']:.2e}", 'F1 Teste': f"{variancias['F1 Teste']:.2e}",
    'Precisão Teste': f"{variancias['Precisão Teste']:.2e}", 'Revocação Teste': f"{variancias['Revocação Teste']:.2e}",
    'Tempo (s)': f"{variancias['Tempo (s)']:.2f}"
}
linha_ic = {
    'Fold': '**IC 95%**', 'Exemplos Treino': '-', 'Exemplos Teste': '-',
    'Perda Treino': f"[{medias['Perda Treino']-ic_95['Perda Treino']:.4f}, {medias['Perda Treino']+ic_95['Perda Treino']:.4f}]",
    'Perda Teste': f"[{medias['Perda Teste']-ic_95['Perda Teste']:.4f}, {medias['Perda Teste']+ic_95['Perda Teste']:.4f}]",
    'Acurácia Teste': f"[{(medias['Acurácia Teste']-ic_95['Acurácia Teste'])*100:.2f}%, {(medias['Acurácia Teste']+ic_95['Acurácia Teste'])*100:.2f}%]",
    'F1 Teste': f"[{medias['F1 Teste']-ic_95['F1 Teste']:.4f}, {medias['F1 Teste']+ic_95['F1 Teste']:.4f}]",
    'Precisão Teste': f"[{medias['Precisão Teste']-ic_95['Precisão Teste']:.4f}, {medias['Precisão Teste']+ic_95['Precisão Teste']:.4f}]",
    'Revocação Teste': f"[{medias['Revocação Teste']-ic_95['Revocação Teste']:.4f}, {medias['Revocação Teste']+ic_95['Revocação Teste']:.4f}]",
    'Tempo (s)': f"[{medias['Tempo (s)']-ic_95['Tempo (s)']:.1f}, {medias['Tempo (s)']+ic_95['Tempo (s)']:.1f}]"
}

df_tabela_final = pd.concat([df_q7_apresentacao, pd.DataFrame([linha_media, linha_dp, linha_var, linha_ic])], ignore_index=True)

display(Markdown('### Tabela 9 — Resultados da Validação Cruzada 10-Fold (Rede Campeã)'))
display(df_tabela_final)

# Gráfico 1: Curvas de Validação ao longo das épocas nos 10 Folds
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
cores_k = plt.cm.tab10.colors

for idx, h_fold in enumerate(q7_historicos_folds):
    c = cores_k[idx % len(cores_k)]
    axes[0].plot(h_fold['val_loss'], color=c, alpha=0.55, linewidth=1.4)
    axes[1].plot(h_fold['val_accuracy'], color=c, alpha=0.55, linewidth=1.4)

# Média geral entre os folds
media_loss_val = pd.concat([h['val_loss'] for h in q7_historicos_folds], axis=1).mean(axis=1)
media_acc_val = pd.concat([h['val_accuracy'] for h in q7_historicos_folds], axis=1).mean(axis=1)

axes[0].plot(media_loss_val, color='black', linewidth=2.5, label='Média dos 10 Folds')
axes[0].set_title('Perda de Teste por Fold ao Longo das Épocas', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Entropia Cruzada'); axes[0].grid(alpha=0.3); axes[0].legend()

axes[1].plot(media_acc_val, color='black', linewidth=2.5, label='Média dos 10 Folds')
axes[1].set_title('Acurácia de Teste por Fold ao Longo das Épocas', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Acurácia'); axes[1].grid(alpha=0.3); axes[1].legend(loc='lower right')

plt.tight_layout()
fig.savefig('q7_curvas_folds.png', dpi=150)
fig.savefig('resultados_q5q6q7/q7_curvas_folds.png', dpi=150)
plt.show()

# Gráfico 2: Dispersão das Métricas e Matriz de Confusão Consolidada
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

# Boxplot das métricas
dados_box = [df_q7['Acurácia Teste'] * 100, df_q7['F1 Teste'] * 100,
             df_q7['Precisão Teste'] * 100, df_q7['Revocação Teste'] * 100]
rotulos_box = ['Acurácia (%)', 'F1-Score (×100)', 'Precisão (×100)', 'Revocação (×100)']

bp = axes[0].boxplot(dados_box, patch_artist=True, tick_labels=rotulos_box)
cores_box = ['lightblue', 'lightgreen', 'lightcoral', 'wheat']
for patch, cor in zip(bp['boxes'], cores_box):
    patch.set_facecolor(cor)
axes[0].set_title('Distribuição e Dispersão das Métricas entre os 10 Folds', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Pontuação (%)')
axes[0].grid(axis='y', alpha=0.3)
for i, d in enumerate(dados_box):
    axes[0].plot(np.random.normal(i + 1, 0.04, size=len(d)), d, 'o', color='darkblue', alpha=0.6, markersize=5)

# Matriz de Confusão Consolidada Out-Of-Fold (Todas as 13.611 instâncias)
cm_oof = confusion_matrix(Y, oof_predicoes, normalize='true')
im_oof = axes[1].imshow(cm_oof, cmap='Blues', vmin=0, vmax=1)
axes[1].set_xticks(range(n_classes), nomes_classes, rotation=45, ha='right', fontsize=9)
axes[1].set_yticks(range(n_classes), nomes_classes, fontsize=9)
axes[1].set_title(f'Matriz de Confusão Consolidada (10-Fold OOF — N = {len(Y):,})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Classe Prevista'); axes[1].set_ylabel('Classe Real')
for r_i in range(n_classes):
    for c_j in range(n_classes):
        axes[1].text(c_j, r_i, f"{cm_oof[r_i, c_j]:.2f}", ha='center', va='center',
                     color='white' if cm_oof[r_i, c_j] > 0.5 else 'black', fontsize=8)
plt.colorbar(im_oof, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig('q7_metricas_dispersao.png', dpi=150)
fig.savefig('resultados_q5q6q7/q7_metricas_dispersao.png', dpi=150)
fig.savefig('q7_matriz_confusao_consolidada.png', dpi=150)
fig.savefig('resultados_q5q6q7/q7_matriz_confusao_consolidada.png', dpi=150)
plt.show()

print("\n=== RESUMO ESTATÍSTICO CONSOLIDADO (10-FOLD CV) ===")
print(f"  Acurácia Média:  {medias['Acurácia Teste']*100:.2f}% ± {desvios['Acurácia Teste']*100:.2f}% (Variância: {variancias['Acurácia Teste']:.2e}) | IC 95%: {linha_ic['Acurácia Teste']}")
print(f"  Medida-F Média:  {medias['F1 Teste']:.4f} ± {desvios['F1 Teste']:.4f} (Variância: {variancias['F1 Teste']:.2e}) | IC 95%: {linha_ic['F1 Teste']}")
print(f"  Perda Média:     {medias['Perda Teste']:.4f} ± {desvios['Perda Teste']:.4f} (Variância: {variancias['Perda Teste']:.2e})")
print(f"  Coef. Variação:  Acurácia = {cv_percent['Acurácia Teste']:.2f}%, F1 = {cv_percent['F1 Teste']:.2f}%, Perda = {cv_percent['Perda Teste']:.2f}%")
''')

relato('relato_q7.md')


# ---------------------------------------------------------------------------------------------
# Mecanismo de Execução no Kernel e Montagem do Notebook teste.ipynb
# ---------------------------------------------------------------------------------------------
def merge_streams(outputs):
    merged = []
    for o in outputs:
        if o['output_type'] == 'stream' and merged and merged[-1]['output_type'] == 'stream' \
                and merged[-1]['name'] == o['name']:
            merged[-1]['text'] += o['text']
        else:
            merged.append(o)
    return merged


def executar():
    from jupyter_client.manager import start_new_kernel
    nb = nbformat.read(NB_PATH, as_version=4)
    # Selecionamos as células base (anteriores a q3q4 e q5q6q7)
    base = [c for c in nb.cells if TAG not in c.get('metadata', {}).get('tags', []) and 'q3q4' not in c.get('metadata', {}).get('tags', [])]
    km, kc = start_new_kernel(kernel_name='python3', cwd=PROJ)

    def run(src, keep=True):
        outputs = []
        def hook(msg):
            t = msg['msg_type']
            if t not in ('stream', 'display_data', 'execute_result', 'error'):
                return
            if t == 'stream':
                if msg['content']['name'] == 'stderr':
                    return
                sys.stdout.write(msg['content']['text']); sys.stdout.flush()
            if t == 'error':
                print('\n'.join(msg['content']['traceback']))
            outputs.append(nbformat.v4.output_from_msg(msg))

        reply = kc.execute_interactive(src, output_hook=hook, timeout=None, allow_stdin=False)
        if reply['content']['status'] != 'ok':
            raise RuntimeError('Falha ao executar a célula:\n' + src[:300])
        return merge_streams(outputs)

    try:
        run("import os, warnings; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; warnings.filterwarnings('ignore')\n"
            "%matplotlib inline\nimport sys; print('Kernel Python:', sys.executable)")
        for i in (0, 1, 3, 4, 5):
            run(base[i].source)
        run(PRELUDE_EXTRA)

        exec_count = max([c.get('execution_count') or 0 for c in nb.cells if c.cell_type == 'code'] + [0])
        executadas = []
        for c in CELLS:
            if c['type'] == 'code':
                exec_count += 1
                print(f'\n>>> executando célula {exec_count}')
                outs = run(c['src'])
                executadas.append({'type': 'code', 'src': c['src'], 'count': exec_count,
                                   'outputs': [dict(o) for o in outs]})
            else:
                executadas.append(c)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(CACHE, 'w', encoding='utf-8') as f:
            json.dump(executadas, f, ensure_ascii=False)
    finally:
        kc.stop_channels(); km.shutdown_kernel(now=True)


def montar_notebook():
    with open(CACHE, encoding='utf-8') as f:
        executadas = json.load(f)
    nb = nbformat.read(NB_PATH, as_version=4)
    # Remove qualquer célula com a tag q5q6q7 já existente
    nb.cells = [c for c in nb.cells if TAG not in c.get('metadata', {}).get('tags', [])]
    meta = {'tags': [TAG]}
    for c in executadas:
        if c['type'] == 'markdown':
            nb.cells.append(nbformat.v4.new_markdown_cell(c['src'], metadata=meta))
        elif c['type'] == 'relato':
            p = os.path.join(OUT_DIR, c['file'])
            if os.path.exists(p):
                with open(p, encoding='utf-8') as f:
                    nb.cells.append(nbformat.v4.new_markdown_cell(f.read().strip(), metadata=meta))
        else:
            cell = nbformat.v4.new_code_cell(c['src'], metadata=meta, execution_count=c['count'])
            cell.outputs = [nbformat.from_dict(o) for o in c['outputs']]
            nb.cells.append(cell)
    nbformat.write(nb, NB_PATH)
    print(f'teste.ipynb atualizado com sucesso: {len(nb.cells)} células totais.')


SMOKE_SUBS = [
    ('MAX_EPOCHS_Q5 = 80', 'MAX_EPOCHS_Q5 = 2'),
    ('SEEDS_Q5 = [0, 1, 2]', 'SEEDS_Q5 = [0]'),
    ('MAX_EPOCHS_Q6 = 100', 'MAX_EPOCHS_Q6 = 2'),
    ('SEEDS_Q6 = [0, 1, 2]', 'SEEDS_Q6 = [0]'),
    ('K_FOLDS = 10', 'K_FOLDS = 2'),
    ('EPOCHS_Q7 = 80', 'EPOCHS_Q7 = 2')
]

if __name__ == '__main__':
    if '--smoke' in sys.argv:
        CACHE = os.path.join(OUT_DIR, 'celulas_smoke.json')
        for c in CELLS:
            if c['type'] == 'code':
                for a, b in SMOKE_SUBS:
                    c['src'] = c['src'].replace(a, b)
        executar()
    else:
        if '--rebuild' not in sys.argv:
            executar()
        montar_notebook()
