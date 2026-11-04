"""
Questões 3 e 4 — Topologia da rede e Influência dos dados de treinamento.

Executa as células abaixo num kernel Jupyter real (mesmo pré-processamento das células 0-5 do
notebook), grava as saídas reais e anexa as células ao final de `teste.ipynb`.

Uso:
    py run_q3_q4.py            # executa tudo (~40 min em CPU) e atualiza o notebook
    py run_q3_q4.py --rebuild  # só remonta o notebook a partir do cache (ex.: após editar os relatos)

As células anexadas ficam marcadas com a tag "q3q4"; rodar de novo substitui as anteriores.
Os relatos em markdown vêm de resultados_q3q4/relato_q3.md e relato_q4.md (se existirem).
"""
import os
import sys
import json
import nbformat

PROJ = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(PROJ, 'teste.ipynb')
OUT_DIR = os.path.join(PROJ, 'resultados_q3q4')
CACHE = os.path.join(OUT_DIR, 'celulas_q3q4.json')
TAG = 'q3q4'

# ---------------------------------------------------------------------------------------------
# Código executado antes das células (não vai para o notebook: equivale às células 0-5, 7 e 15)
# ---------------------------------------------------------------------------------------------
PRELUDE_EXTRA = r'''
tf.get_logger().setLevel('ERROR')   # silencia avisos de "retracing" do tf.function

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

# ---------------------------------------------------------------------------------------------
# Células da Questão 3
# ---------------------------------------------------------------------------------------------
CELLS = []
md = lambda s: CELLS.append({'type': 'markdown', 'src': s.strip('\n')})
code = lambda s: CELLS.append({'type': 'code', 'src': s.strip('\n')})
relato = lambda f: CELLS.append({'type': 'relato', 'file': f})

md(r'''
# 3. Topologia da Rede — Influência do Número de Camadas e Neurônios por Camada

Mantendo fixos os hiperparâmetros escolhidos na Questão 2 ($\eta^* = 0.01$, $\alpha^* = 0.5$, SGD, lote 32),
variamos sistematicamente a topologia da MLP:

* **Grade uniforme:** 1, 2 e 3 camadas ocultas × 10 a 100 neurônios por camada (incrementos de 10) → 30 topologias;
* **Topologias em funil (complementares):** 64-32 (base da Q1), 60-30-15, 80-40-20, 100-50 e 100-50-25.

ReLU nas camadas ocultas, Softmax na saída e Entropia Cruzada Categórica. **Critério de parada:** perda de treino
$\le 0.001$ ou 100 épocas. A perda é calculada no treino; o conjunto de **validação** acompanha a generalização
(underfitting/overfitting) e a **Medida-F (macro)**. O conjunto de **teste não é usado** nesta etapa.
''')

code(r'''
import os, time
import matplotlib.patheffects as pe
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
os.makedirs('resultados_q3q4', exist_ok=True)

ETA, ALPHA = 0.01, 0.5        # (η*, α*) = melhor combinação da Questão 2
BATCH_SIZE = 32
MAX_EPOCHS_Q3 = 100
STOP_LOSS = 0.001
Q3_SEED = 42
N_LAYERS = [1, 2, 3]
N_NEURONS = list(range(10, 101, 10))
PIRAMIDAIS = [[64, 32], [100, 50], [60, 30, 15], [80, 40, 20], [100, 50, 25]]
LIMIAR_UNDERFIT = 0.20        # perda (treino E validação) acima disso ≈ acurácia < ~92,5%

def build_topology(hidden_units, seed, lr=ETA, momentum=ALPHA):
    """MLP ReLU/Softmax com a topologia pedida e pesos iniciais determinados pela semente."""
    reset_seeds(seed)
    model = keras.Sequential([keras.Input(shape=(n_features,))])
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, activation='relu',
                               kernel_initializer=keras.initializers.HeNormal(seed=seed + i),
                               name=f'oculta_{i+1}'))
    model.add(layers.Dense(n_classes, activation='softmax',
                           kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 100),
                           name='saida'))
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
                  loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'],
                  steps_per_execution=32)
    return model

def metricas(model, X, y):
    """Perda, acurácia, precisão, revocação e Medida-F (média macro sobre as 7 classes)."""
    prob = model.predict(X, verbose=0, batch_size=1024)
    pred = prob.argmax(axis=1)
    return dict(loss=log_loss(y, prob, labels=range(n_classes)),
                acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, average='macro', zero_division=0),
                rec=recall_score(y, pred, average='macro', zero_division=0),
                f1=f1_score(y, pred, average='macro'),
                rec_classe=recall_score(y, pred, average=None, labels=range(n_classes), zero_division=0))

def nome(hu):  return '-'.join(map(str, hu))
def parse(s):  return [int(v) for v in s.split('-')]

def treinar_topologia(hidden_units, seed=Q3_SEED, epochs=MAX_EPOCHS_Q3):
    model = build_topology(hidden_units, seed)
    cb = EarlyStoppingLossThreshold(STOP_LOSS)
    t0 = time.perf_counter()
    h = model.fit(X_tr_s, Y_tr, validation_data=(X_val_s, Y_val), epochs=epochs,
                  batch_size=BATCH_SIZE, callbacks=[cb], shuffle=True, verbose=0)
    tempo = time.perf_counter() - t0
    hist = pd.DataFrame(h.history); hist.index += 1
    mv = metricas(model, X_val_s, y_val)
    uniforme = len(set(hidden_units)) == 1
    linha = {
        'Topologia': nome(hidden_units),
        'Tipo': 'uniforme' if uniforme else 'funil',
        'Camadas': len(hidden_units),
        'Neurônios/camada': hidden_units[0] if uniforme else np.nan,
        'Parâmetros': model.count_params(),
        'Perda treino': hist['loss'].iloc[-1],
        'Acurácia treino': hist['accuracy'].iloc[-1],
        'Perda val': hist['val_loss'].iloc[-1],
        'Acurácia val': hist['val_accuracy'].iloc[-1],
        'Menor perda val': hist['val_loss'].min(),
        'Época menor perda val': int(hist['val_loss'].idxmin()),
        'Épocas': len(hist),
        'Atingiu perda ≤ 0.001': cb.converged_epoch is not None,
        'Tempo (s)': tempo,
        'Precisão val': mv['prec'], 'Revocação val': mv['rec'], 'F1 val': mv['f1'],
    }
    return linha, hist, model

def diagnosticar(r):
    subida_val = r['Perda val'] - r['Menor perda val']   # quanto a perda de validação subiu após o mínimo
    gap = r['Perda val'] - r['Perda treino']
    if r['Perda treino'] > LIMIAR_UNDERFIT and r['Perda val'] > LIMIAR_UNDERFIT:
        return 'Underfitting'
    if subida_val > 0.01 or gap > 0.05:
        return 'Overfitting'
    if subida_val > 0.003:
        return 'Início de overfitting'
    return 'Bom ajuste'

print(f"η={ETA}, α={ALPHA}, lote={BATCH_SIZE}, até {MAX_EPOCHS_Q3} épocas ou perda ≤ {STOP_LOSS}")
''')

code(r'''
topologias = [[n] * L for L in N_LAYERS for n in N_NEURONS] + PIRAMIDAIS
q3_rows, q3_hist = [], {}
print(f"Treinando {len(topologias)} topologias...")
for hu in topologias:
    linha, hist, _ = treinar_topologia(hu)
    q3_rows.append(linha); q3_hist[linha['Topologia']] = hist
    print(f"[{linha['Topologia']:>11s}] params={linha['Parâmetros']:>6d} | perda treino={linha['Perda treino']:.4f} "
          f"val={linha['Perda val']:.4f} (mín {linha['Menor perda val']:.4f} @ ép. {linha['Época menor perda val']:>3d}) "
          f"| F1 val={linha['F1 val']:.4f} | {linha['Tempo (s)']:.1f} s")

df_q3 = pd.DataFrame(q3_rows)
df_q3['Diagnóstico'] = df_q3.apply(diagnosticar, axis=1)
df_q3.to_csv('resultados_q3q4/q3_topologias.csv', index=False)
print("\nNenhuma topologia atingiu perda ≤ 0.001:", not df_q3['Atingiu perda ≤ 0.001'].any())
''')

code(r'''
cols = ['Topologia', 'Camadas', 'Parâmetros', 'Perda treino', 'Perda val', 'Menor perda val',
        'Época menor perda val', 'Acurácia treino', 'Acurácia val', 'Precisão val', 'Revocação val',
        'F1 val', 'Tempo (s)', 'Diagnóstico']
display(Markdown('### Tabela 3 — Todas as topologias (ordenadas pela perda final de validação)'))
display(df_q3.sort_values('Perda val')[cols].reset_index(drop=True))
display(Markdown('### Contagem de diagnósticos por número de camadas'))
display(pd.crosstab(df_q3['Camadas'], df_q3['Diagnóstico']))
''')

code(r'''
uni = df_q3[df_q3['Tipo'] == 'uniforme']
fig, axes = plt.subplots(1, 3, figsize=(19, 4.2))
specs = [('Perda val', 'viridis_r', '{:.3f}', 'Perda final de validação'),
         ('F1 val', 'viridis', '{:.3f}', 'Medida-F macro (validação)'),
         ('Tempo (s)', 'magma_r', '{:.0f}', 'Tempo de treinamento (s)')]
for ax, (col, cmap, fmt, titulo) in zip(axes, specs):
    M = uni.pivot(index='Camadas', columns='Neurônios/camada', values=col)
    im = ax.imshow(M.values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(M.shape[1]), [int(c) for c in M.columns])
    ax.set_yticks(range(M.shape[0]), [f'{l} camada(s)' for l in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M.values[i, j]), ha='center', va='center', fontsize=8, color='white',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    ax.set(title=titulo, xlabel='Neurônios por camada')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig('q3_heatmaps.png', dpi=150)
plt.show()
''')

code(r'''
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
cores_L = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
for L, g in uni.groupby('Camadas'):
    g = g.sort_values('Neurônios/camada')
    axes[0, 0].plot(g['Neurônios/camada'], g['Perda treino'], 'o-', color=cores_L[L], label=f'{L} camada(s) – treino')
    axes[0, 0].plot(g['Neurônios/camada'], g['Perda val'], 's--', color=cores_L[L], alpha=.8, label=f'{L} camada(s) – validação')
    axes[0, 1].plot(g['Neurônios/camada'], g['Perda val'] - g['Menor perda val'], 'o-', color=cores_L[L], label=f'{L} camada(s)')
    axes[1, 0].plot(g['Neurônios/camada'], g['F1 val'], 'o-', color=cores_L[L], label=f'{L} camada(s)')
for L, g in df_q3.groupby('Camadas'):
    axes[1, 1].scatter(g['Parâmetros'], g['Tempo (s)'], color=cores_L[L], label=f'{L} camada(s)',
                       marker='o', s=40)
fun = df_q3[df_q3['Tipo'] == 'funil']
axes[1, 1].scatter(fun['Parâmetros'], fun['Tempo (s)'], facecolors='none', edgecolors='black', s=120, label='funil')
for _, r in fun.iterrows():
    axes[1, 1].annotate(r['Topologia'], (r['Parâmetros'], r['Tempo (s)']), fontsize=7, xytext=(4, 4), textcoords='offset points')
axes[0, 0].set(title='Perda final: treino (—) e validação (--)', xlabel='Neurônios por camada', ylabel='Entropia cruzada')
axes[0, 1].set(title='Subida da perda de validação após o seu mínimo (indício de overfitting)',
               xlabel='Neurônios por camada', ylabel='Perda val final − menor perda val')
axes[1, 0].set(title='Medida-F macro na validação', xlabel='Neurônios por camada', ylabel='F1 macro')
axes[1, 1].set(title='Custo computacional', xlabel='Nº de parâmetros (escala log)', ylabel='Tempo de treino (s)', xscale='log')
for ax in axes.ravel(): ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig('q3_tendencias.png', dpi=150)
plt.show()
''')

code(r'''
top2 = df_q3.nsmallest(2, 'Perda val')['Topologia'].tolist()
exemplos = list(dict.fromkeys(['10', '100', '10-10-10', '100-100-100'] + top2))
fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
for ax, t in zip(axes.ravel(), exemplos):
    h = q3_hist[t]
    ax.plot(h['loss'], label='treino'); ax.plot(h['val_loss'], label='validação', linestyle='--')
    ep = h['val_loss'].idxmin()
    ax.axvline(ep, color='gray', linestyle=':', label=f'mín. val (ép. {ep})')
    diag = df_q3.set_index('Topologia').loc[t, 'Diagnóstico']
    ax.set(title=f'{t}  —  {diag}', xlabel='Época', ylabel='Entropia cruzada', ylim=(0.10, 0.45))
    ax.grid(alpha=.3); ax.legend(fontsize=8)
for ax in axes.ravel()[len(exemplos):]: ax.axis('off')
plt.suptitle('Curvas de convergência (treino × validação) para topologias representativas', y=1.0)
plt.tight_layout()
fig.savefig('q3_curvas_loss.png', dpi=150)
plt.show()
''')

md(r'''
### Estabilidade das melhores topologias

As 6 topologias com menor perda de validação foram re-treinadas com mais 3 sementes (0, 1 e 2), totalizando
4 inicializações cada. As **quatro melhores** são escolhidas pela menor perda média de validação, observando
também a Medida-F e o desvio padrão entre sementes.
''')

code(r'''
N_CANDIDATAS, SEEDS_ESTAB = 6, [0, 1, 2]
candidatas = df_q3.nsmallest(N_CANDIDATAS, 'Perda val')
est_rows = []
for _, c in candidatas.iterrows():
    est_rows.append({**c.to_dict(), 'Semente': Q3_SEED})
    for s in SEEDS_ESTAB:
        linha, _, _ = treinar_topologia(parse(c['Topologia']), seed=s)
        est_rows.append({**linha, 'Semente': s})
        print(f"[{c['Topologia']:>11s}] semente {s}: perda val={linha['Perda val']:.4f}  F1 val={linha['F1 val']:.4f}")
df_estab = pd.DataFrame(est_rows)
df_estab.to_csv('resultados_q3q4/q3_estabilidade.csv', index=False)

q3_rank = (df_estab.groupby('Topologia')
           .agg(**{'Parâmetros': ('Parâmetros', 'first'),
                   'Perda treino (média)': ('Perda treino', 'mean'),
                   'Perda val (média)': ('Perda val', 'mean'), 'Perda val (dp)': ('Perda val', 'std'),
                   'Acurácia val (média)': ('Acurácia val', 'mean'),
                   'F1 val (média)': ('F1 val', 'mean'), 'F1 val (dp)': ('F1 val', 'std'),
                   'Precisão val (média)': ('Precisão val', 'mean'), 'Revocação val (média)': ('Revocação val', 'mean'),
                   'Tempo médio (s)': ('Tempo (s)', 'mean')})
           .sort_values(['Perda val (média)', 'F1 val (média)'], ascending=[True, False]))
q3_rank.insert(0, 'Posição', range(1, len(q3_rank) + 1))
q3_rank.to_csv('resultados_q3q4/q3_ranking_final.csv')
finalistas = q3_rank.head(4)
MELHOR_TOPOLOGIA = parse(finalistas.index[0])
display(Markdown('### Tabela 4 — Estabilidade (4 sementes) e seleção das quatro melhores topologias'))
display(q3_rank)
print('Quatro topologias selecionadas:', list(finalistas.index))
print('Melhor topologia (usada na Questão 4):', MELHOR_TOPOLOGIA)
''')

code(r'''
ordem = q3_rank.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, col, titulo in [(axes[0], 'Perda val', 'Perda final de validação'), (axes[1], 'F1 val', 'Medida-F macro (validação)')]:
    dados = [df_estab.loc[df_estab['Topologia'] == t, col].values for t in ordem]
    ax.boxplot(dados, showmeans=True)
    ax.set_xticks(range(1, len(ordem) + 1), [('★ ' if t in finalistas.index else '') + t for t in ordem], rotation=20)
    ax.set(title=f'{titulo} — 4 sementes por topologia', ylabel=col); ax.grid(alpha=.3)
plt.tight_layout()
fig.savefig('q3_estabilidade.png', dpi=150)
plt.show()
''')

relato('relato_q3.md')

# ---------------------------------------------------------------------------------------------
# Células da Questão 4
# ---------------------------------------------------------------------------------------------
md(r'''
# 4. Influência dos Dados de Treinamento — Qualidade e Quantidade de Exemplos

Usamos a **melhor topologia da Questão 3** com os mesmos hiperparâmetros ($\eta = 0.01$, $\alpha = 0.5$, SGD, lote 32,
ReLU/Softmax, entropia cruzada) e variamos os dados de treinamento:

* **Quantidade:** 5%, 10%, 20%, 40%, 60%, 80% e 100% do conjunto de treinamento (9.208 exemplos), com
  **amostragem estratificada** (proporção das classes preservada). Cada tamanho é repetido com 3 sementes
  (amostra e pesos iniciais diferentes).
* **Qualidade / representatividade:** (i) amostra de 20% estratificada × aleatória simples × **enviesada**
  (classe BARBUNYA reduzida a 10% da sua cota); (ii) **ruído de rótulos** — 10%, 20% e 30% dos exemplos de treino
  com a classe trocada aleatoriamente.

Para comparar tamanhos de forma justa, cada treino vai **até convergir**: parada quando a perda de treino
$\le 0.001$ ou quando a perda de validação não melhora por 20 épocas (máx. 300), restaurando os melhores pesos.
O padronizador (`StandardScaler`) é ajustado só na amostra usada. A validação e o teste (2.709 exemplos, nunca
usados para treino ou escolha de parâmetros) são sempre os mesmos conjuntos completos e com rótulos corretos.
''')

code(r'''
FRACOES = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS_Q4 = [0, 1, 2]
MAX_EPOCHS_Q4, PACIENCIA = 300, 20
print('Topologia usada:', nome(MELHOR_TOPOLOGIA))

def subamostrar(frac, seed, estratificado=True):
    idx = np.arange(len(y_tr))
    if frac >= 1.0:
        return idx
    sel, _ = train_test_split(idx, train_size=frac, random_state=seed,
                              stratify=y_tr if estratificado else None)
    return sel

def treinar_q4(idx, seed, y_treino=None):
    """Treina a melhor topologia apenas com os exemplos idx (rótulos y_treino, se dados)."""
    y_sub = y_tr[idx] if y_treino is None else y_treino[idx]
    sc = StandardScaler().fit(X_tr[idx])
    Xs, Xv, Xt = sc.transform(X_tr[idx]), sc.transform(X_val), sc.transform(X_test)
    model = build_topology(MELHOR_TOPOLOGIA, seed)
    cbs = [EarlyStoppingLossThreshold(STOP_LOSS),
           keras.callbacks.EarlyStopping(monitor='val_loss', patience=PACIENCIA, restore_best_weights=True)]
    t0 = time.perf_counter()
    h = model.fit(Xs, keras.utils.to_categorical(y_sub, n_classes), validation_data=(Xv, Y_val),
                  epochs=MAX_EPOCHS_Q4, batch_size=BATCH_SIZE, callbacks=cbs, shuffle=True, verbose=0)
    tempo = time.perf_counter() - t0
    hist = pd.DataFrame(h.history); hist.index += 1
    tr, va, te = metricas(model, Xs, y_sub), metricas(model, Xv, y_val), metricas(model, Xt, y_test)
    return {
        'Exemplos': len(idx), 'Semente': seed,
        'Épocas treinadas': len(hist), 'Época de convergência': int(hist['val_loss'].idxmin()),
        'Tempo (s)': tempo, 'Tempo/época (s)': tempo / len(hist),
        'Perda treino': tr['loss'], 'Perda val': va['loss'], 'Perda teste': te['loss'],
        'Acurácia treino': tr['acc'], 'Acurácia val': va['acc'], 'Acurácia teste': te['acc'],
        'F1 treino': tr['f1'], 'F1 val': va['f1'], 'F1 teste': te['f1'],
        'Precisão teste': te['prec'], 'Revocação teste': te['rec'],
        'rec_val': va['rec_classe'], 'rec_teste': te['rec_classe'],
    }, hist
''')

code(r'''
q4_rows, q4_hist = [], {}
for frac in FRACOES:
    for s in SEEDS_Q4:
        r, h = treinar_q4(subamostrar(frac, s), s)
        r['Fração'] = frac
        q4_rows.append(r); q4_hist[(frac, s)] = h
        print(f"{frac:>5.0%} ({r['Exemplos']:>4d} ex.) semente {s}: épocas={r['Épocas treinadas']:>3d} "
              f"| perda tr/val/teste={r['Perda treino']:.4f}/{r['Perda val']:.4f}/{r['Perda teste']:.4f} "
              f"| F1 val={r['F1 val']:.4f} teste={r['F1 teste']:.4f} | {r['Tempo (s)']:.1f} s")
df_q4 = pd.DataFrame(q4_rows)
df_q4.drop(columns=['rec_val', 'rec_teste']).to_csv('resultados_q3q4/q4_quantidade.csv', index=False)

met_q4 = ['Exemplos', 'Épocas treinadas', 'Época de convergência', 'Tempo (s)', 'Perda treino', 'Perda val',
          'Perda teste', 'Acurácia treino', 'Acurácia val', 'Acurácia teste', 'F1 val', 'F1 teste']
q4_media = df_q4.groupby('Fração')[met_q4].mean()
q4_dp = df_q4.groupby('Fração')[met_q4].std()
q4_tabela = pd.DataFrame(index=[f'{f:.0%}' for f in q4_media.index])
for c in met_q4:
    casas = 0 if c in ('Exemplos',) else (1 if c in ('Épocas treinadas', 'Época de convergência', 'Tempo (s)') else 4)
    q4_tabela[c] = [f'{m:.{casas}f} ± {d:.{casas}f}' if c != 'Exemplos' else f'{m:.0f}'
                    for m, d in zip(q4_media[c], q4_dp[c])]
q4_media.to_csv('resultados_q3q4/q4_quantidade_media.csv')
display(Markdown('### Tabela 5 — Desempenho por tamanho do conjunto de treinamento (média ± desvio, 3 sementes)'))
display(q4_tabela)
''')

code(r'''
x = q4_media['Exemplos'].values.round().astype(int)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, met, titulo in [(axes[0, 0], 'Perda', 'Perda (entropia cruzada)'),
                        (axes[0, 1], 'Acurácia', 'Acurácia'),
                        (axes[1, 0], 'F1', 'Medida-F macro')]:
    for conj, cor, mk in [('treino', 'tab:blue', 'o'), ('val', 'tab:orange', 's'), ('teste', 'tab:green', '^')]:
        col = f'{met} {conj}'
        if col not in df_q4: continue
        g = df_q4.groupby('Fração')[col]
        ax.errorbar(x, g.mean(), yerr=g.std(), marker=mk, color=cor, capsize=3, label=conj)
    ax.set(title=f'Curva de generalização — {titulo}', xlabel='Exemplos de treinamento (escala log)', xscale='log')
    ax.minorticks_off()
    ax.set_xticks(x, [f'{n}\n({f:.0%})' for n, f in zip(x, q4_media.index)], fontsize=8)
    ax.grid(alpha=.3); ax.legend()
g = df_q4.groupby('Fração')
axes[1, 1].errorbar(x, g['Tempo (s)'].mean(), yerr=g['Tempo (s)'].std(), marker='o', capsize=3, label='tempo total (s)')
ax2 = axes[1, 1].twinx()
ax2.errorbar(x, g['Época de convergência'].mean(), yerr=g['Época de convergência'].std(), marker='s', color='tab:red',
             capsize=3, label='época de convergência')
axes[1, 1].set(title='Custo computacional', xlabel='Exemplos de treinamento (escala log)', ylabel='Tempo (s)', xscale='log')
axes[1, 1].minorticks_off()
axes[1, 1].set_xticks(x, [f'{n}\n({f:.0%})' for n, f in zip(x, q4_media.index)], fontsize=8)
ax2.set_ylabel('Época de menor perda de validação', color='tab:red')
axes[1, 1].grid(alpha=.3); axes[1, 1].legend(loc='upper left'); ax2.legend(loc='upper center')
plt.tight_layout()
fig.savefig('q4_curvas_generalizacao.png', dpi=150)
plt.show()
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
cores_f = plt.cm.viridis(np.linspace(0, .9, len(FRACOES)))
for cor, frac in zip(cores_f, FRACOES):
    h = q4_hist[(frac, 0)]
    axes[0].plot(h['loss'], color=cor, label=f'{frac:.0%}')
    axes[1].plot(h['val_loss'], color=cor, label=f'{frac:.0%}')
axes[0].set(title='Perda de treino por época (semente 0)', xlabel='Época', ylabel='Entropia cruzada', ylim=(0, 0.6))
axes[1].set(title='Perda de validação por época (semente 0)', xlabel='Época')
for ax in axes: ax.grid(alpha=.3); ax.legend(title='Fração', fontsize=8)
plt.tight_layout()
fig.savefig('q4_curvas_epocas.png', dpi=150)
plt.show()
''')

md(r'''
### Qualidade e representatividade dos exemplos
''')

code(r'''
CLASSE_SUB = 'BARBUNYA'
cod_sub = [k for k, v in classes.items() if v == CLASSE_SUB][0]

def amostra_enviesada(frac, seed, manter=0.10):
    """Amostra estratificada em que a classe CLASSE_SUB fica com só `manter` da sua cota."""
    idx = subamostrar(frac, seed, estratificado=True)
    rng = np.random.default_rng(seed)
    alvo = idx[y_tr[idx] == cod_sub]
    keep = rng.choice(alvo, max(1, int(round(manter * len(alvo)))), replace=False)
    return np.concatenate([idx[y_tr[idx] != cod_sub], keep])

def corromper_rotulos(y, taxa, seed):
    """Troca a classe de uma fração `taxa` dos exemplos por outra classe sorteada (sempre diferente)."""
    rng = np.random.default_rng(seed); y2 = y.copy()
    n = int(round(taxa * len(y))); pos = rng.choice(len(y), n, replace=False)
    y2[pos] = (y2[pos] + rng.integers(1, n_classes, n)) % n_classes
    return y2

q4q_rows = []
cenarios = [('20% estratificada', lambda s: subamostrar(0.20, s, True)),
            ('20% aleatória simples', lambda s: subamostrar(0.20, s, False)),
            (f'20% enviesada ({CLASSE_SUB} a 10%)', lambda s: amostra_enviesada(0.20, s))]
for nome_c, f in cenarios:
    for s in SEEDS_Q4:
        idx = f(s)
        r, _ = treinar_q4(idx, s)
        r.update({'Experimento': 'Representatividade', 'Cenário': nome_c,
                  f'% {CLASSE_SUB} na amostra': (y_tr[idx] == cod_sub).mean(),
                  f'Revocação {CLASSE_SUB} (teste)': r['rec_teste'][cod_sub]})
        q4q_rows.append(r)
        print(f"{nome_c:<32s} s={s}: n={r['Exemplos']:>4d} | F1 val={r['F1 val']:.4f} teste={r['F1 teste']:.4f} "
              f"| revocação {CLASSE_SUB} teste={r['rec_teste'][cod_sub]:.3f}")

for taxa in [0.0, 0.10, 0.20, 0.30]:
    for s in SEEDS_Q4:
        if taxa == 0.0:
            r = dict(df_q4[(df_q4['Fração'] == 1.0) & (df_q4['Semente'] == s)].iloc[0])
        else:
            r, _ = treinar_q4(np.arange(len(y_tr)), s, y_treino=corromper_rotulos(y_tr, taxa, s))
        r.update({'Experimento': 'Ruído de rótulos', 'Cenário': f'{taxa:.0%} de rótulos trocados',
                  'Taxa de ruído': taxa, f'Revocação {CLASSE_SUB} (teste)': r['rec_teste'][cod_sub]})
        q4q_rows.append(r)
        print(f"ruído {taxa:>4.0%} s={s}: acc treino (rótulos ruidosos)={r['Acurácia treino']:.4f} "
              f"| F1 val={r['F1 val']:.4f} teste={r['F1 teste']:.4f} | épocas={r['Épocas treinadas']}")

df_q4q = pd.DataFrame(q4q_rows)
df_q4q.drop(columns=['rec_val', 'rec_teste']).to_csv('resultados_q3q4/q4_qualidade.csv', index=False)
met_q = ['Exemplos', 'Épocas treinadas', 'Tempo (s)', 'Acurácia treino', 'Perda val', 'Acurácia val', 'F1 val',
         'Perda teste', 'Acurácia teste', 'F1 teste', f'Revocação {CLASSE_SUB} (teste)']
q4q_media = df_q4q.groupby(['Experimento', 'Cenário'], sort=False)[met_q].mean()
q4q_dp = df_q4q.groupby(['Experimento', 'Cenário'], sort=False)[met_q].std()
q4q_media.to_csv('resultados_q3q4/q4_qualidade_media.csv')
display(Markdown('### Tabela 6 — Qualidade dos exemplos (média de 3 sementes)'))
display(q4q_media)
display(Markdown('Desvio padrão:'))
display(q4q_dp)
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
rep = df_q4q[df_q4q['Experimento'] == 'Representatividade']
cen = rep['Cenário'].unique()
w = 0.27
for k, (col, cor) in enumerate([('F1 val', 'tab:orange'), ('F1 teste', 'tab:green'),
                                (f'Revocação {CLASSE_SUB} (teste)', 'tab:purple')]):
    g = rep.groupby('Cenário', sort=False)[col]
    axes[0].bar(np.arange(len(cen)) + (k - 1) * w, g.mean(), w, yerr=g.std(), capsize=3, color=cor, label=col)
axes[0].set_xticks(range(len(cen)), cen, fontsize=9)
axes[0].set(title='Representatividade da amostra (20% dos dados)', ylim=(0.5, 1.0)); axes[0].legend(fontsize=8)
rui = df_q4q[df_q4q['Experimento'] == 'Ruído de rótulos']
for col, cor, mk in [('Acurácia treino', 'tab:blue', 'o'), ('Acurácia val', 'tab:orange', 's'),
                     ('Acurácia teste', 'tab:green', '^'), ('F1 teste', 'tab:red', 'D')]:
    g = rui.groupby('Taxa de ruído')[col]
    axes[1].errorbar(g.mean().index * 100, g.mean(), yerr=g.std(), marker=mk, color=cor, capsize=3, label=col)
axes[1].set(title='Ruído de rótulos no treino (100% dos dados)', xlabel='% de rótulos trocados no treino',
            ylabel='Métrica'); axes[1].legend(fontsize=8)
for ax in axes: ax.grid(alpha=.3)
plt.tight_layout()
fig.savefig('q4_qualidade.png', dpi=150)
plt.show()
''')

relato('relato_q4.md')


# ---------------------------------------------------------------------------------------------
# Execução num kernel Jupyter e montagem do notebook
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
    base = [c for c in nb.cells if TAG not in c.get('metadata', {}).get('tags', [])]
    km, kc = start_new_kernel(kernel_name='python3', cwd=PROJ)

    def run(src, keep=True):
        outputs = []

        def hook(msg):
            t = msg['msg_type']
            if t not in ('stream', 'display_data', 'execute_result', 'error'):
                return
            if t == 'stream':
                if msg['content']['name'] == 'stderr':
                    return  # avisos do TensorFlow não entram no notebook
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
            "%matplotlib inline\nimport sys; print(sys.executable)")
        for i in (0, 1, 3, 4, 5):           # imports, carga, rótulos, duplicatas, divisão/padronização
            run(base[i].source)
        run(PRELUDE_EXTRA)

        exec_count = max([c.get('execution_count') or 0 for c in base if c.cell_type == 'code'] + [0])
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
    print(f'teste.ipynb atualizado: {len(nb.cells)} células.')


SMOKE_SUBS = [('MAX_EPOCHS_Q3 = 100', 'MAX_EPOCHS_Q3 = 3'), ('list(range(10, 101, 10))', '[10, 100]'),
              ('SEEDS_ESTAB = 6, [0, 1, 2]', 'SEEDS_ESTAB = 6, [0]'), ('MAX_EPOCHS_Q4, PACIENCIA = 300, 20', 'MAX_EPOCHS_Q4, PACIENCIA = 4, 2'),
              ('SEEDS_Q4 = [0, 1, 2]', 'SEEDS_Q4 = [0, 1]'), ('[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]', '[0.05, 0.20, 1.00]'),
              ('[0.0, 0.10, 0.20, 0.30]', '[0.0, 0.20]')]

if __name__ == '__main__':
    if '--smoke' in sys.argv:          # teste rápido: poucas épocas, não altera o notebook
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
