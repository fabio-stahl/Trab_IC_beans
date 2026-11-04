### Relato Técnico — Questão 6

#### a. Descrição do experimento
* **Objetivo:** Realizar a **Validação Inicial** comparativa das **quatro melhores topologias de rede neural** selecionadas na Questão 3 diretamente no conjunto de teste reservado ($N = 2.709$ instâncias), o qual permaneceu estritamente intocado durante todo o processo prévio de calibração de hiperparâmetros e arquitetura.
* **Topologias Avaliadas:**
  1. **30-30:** 2 camadas ocultas com 30 neurônios cada (1.657 parâmetros treináveis);
  2. **100:** 1 camada oculta com 100 neurônios (2.407 parâmetros treináveis);
  3. **50:** 1 camada oculta com 50 neurônios (1.207 parâmetros treináveis);
  4. **30:** 1 camada oculta com 30 neurônios (727 parâmetros treináveis).
* **Condições de Treinamento e Avaliação:**
  * **Conjuntos de Dados:** Treinamento conduzido sobre o conjunto de treino completo ($9.208$ exemplos padronizados com `StandardScaler` ajustado apenas no treino) e avaliação no teste reservado ($2.709$ instâncias).
  * **Otimização:** SGD com taxa de aprendizado $\eta = 0.01$, momento $\alpha = 0.5$, tamanho de lote 32, entropia cruzada categórica e 100 épocas de treinamento.
  * **Repetições:** Cada topologia foi executada com 3 sementes de inicialização estocástica (`SEEDS = [0, 1, 2]`) para mitigar o viés do ponto de partida sináptico.
  * **Monitoramento Contínuo:** A cada época registraram-se a perda e acurácia simultaneamente no treino e no teste, mapeando a trajetória da capacidade de generalização e a presença de eventual *overfitting*.

#### b. Resultados obtidos

Os resultados médios obtidos após 100 épocas completas para as quatro arquiteturas são apresentados na tabela a seguir:

**Tabela 8: Comparativo das Quatro Topologias no Conjunto de Teste (Média ± Desvio Padrão, 3 sementes)**

| Topologia | Camadas | Parâmetros | Perda Treino | Perda Teste | Acurácia Teste | F1-Score Teste | Precisão Teste | Revocação Teste | Gap Acurácia (p.p.) | Tempo Médio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **50** | 1 | 1.207 | 0.1808 ± 0.0006 | 0.2043 ± 0.0030 | **92.72% ± 0.20%** | **0.9366 ± 0.0020** | **0.9367 ± 0.0014** | **0.9366 ± 0.0025** | +0.65 p.p. | 40.2 s |
| **100** | 1 | 2.407 | 0.1770 ± 0.0012 | 0.2043 ± 0.0017 | 92.57% ± 0.02% | 0.9355 ± 0.0003 | 0.9363 ± 0.0004 | 0.9349 ± 0.0005 | +0.96 p.p. | 37.0 s |
| **30** | 1 | 727 | 0.1868 ± 0.0023 | 0.2059 ± 0.0019 | 92.58% ± 0.18% | 0.9352 ± 0.0016 | 0.9358 ± 0.0014 | 0.9350 ± 0.0018 | +0.44 p.p. | 37.8 s |
| **30-30** | 2 | 1.657 | 0.1718 ± 0.0021 | 0.2081 ± 0.0045 | 92.46% ± 0.14% | 0.9345 ± 0.0007 | 0.9353 ± 0.0010 | 0.9341 ± 0.0010 | +1.16 p.p. | 42.5 s |

*Figuras associadas:* [`q6_curvas_treino_teste.png`](q6_curvas_treino_teste.png) (Curvas de perda de treino e teste por época para as 4 redes), [`q6_comparativo_top4.png`](q6_comparativo_top4.png) (Barras comparativas de Acurácia e Medida-F no teste) e [`q6_matriz_confusao_teste.png`](q6_matriz_confusao_teste.png) (Matriz de confusão normalizada no conjunto de teste para a rede campeã).

#### c. Análise e discussão
* **Generalização e Desempenho no Teste:** As quatro topologias finalistas demonstraram um comportamento excepcionalmente consistente e parelho no conjunto de teste reservado ($2.709$ instâncias), todas superando a marca de $92.4\%$ de acurácia e $0.934$ de Medida-F macro. A topologia **50** (1 camada oculta com 50 neurônios) conquistou a liderança absoluta em generalização, atingindo **$92.72\% \pm 0.20\%$** de acurácia e **$0.9366 \pm 0.0020$** de Medida-F média.
* **Avaliação de Overfitting / Underfitting (Gap de Generalização):** O gap entre a acurácia de treinamento e teste foi estritamente controlado em todas as configurações, variando de $+0.44$ p.p. na rede de 30 neurônios até $+1.16$ p.p. na rede 30-30. Na topologia campeã (**50**), o gap foi de apenas $+0.65$ p.p., indicando ausência completa de sobreajuste memorizador. As curvas de perda de teste em [`q6_curvas_treino_teste.png`](q6_curvas_treino_teste.png) estabilizam-se assintoticamente sem qualquer inflexão ascendente.
* **Parcimônia e Eficiência Arquitetural:** Com apenas $1.207$ parâmetros treináveis — metade da capacidade da rede de 100 neurônios ($2.407$ parâmetros) —, a rede **50** demonstrou que para 16 variáveis de entrada um espaço latente de 50 dimensões é suficiente e ideal para projetar hiperplanos separadores suaves e não-degenerados.
* **Análise de Erros na Matriz de Confusão:** A matriz de confusão da rede campeã (**50**) no conjunto de teste evidencia que classes morfologicamente bem delimitadas como `BOMBAY` obtêm $99\%$ de acerto. As confusões residuais concentram-se estritamente nas fronteiras biológicas conhecidas (`SIRA` $\leftrightarrow$ `DERMASON` e `BARBUNYA` $\leftrightarrow$ `CALI`), conforme já detectado nas etapas anteriores.

#### d. Conclusão da etapa e Eleição da Campeã
* A arquitetura **50** (1 camada oculta com 50 neurônios, 1.207 parâmetros) é eleita de forma incontestável como a **rede campeã definitiva da Validação Inicial**, obtendo a maior acurácia de teste ($92.72\%$), maior Medida-F ($0.9366$), menor perda no teste ($0.2043$) e excelente regularidade estatística.
* **Decisão para a etapa seguinte:** A rede **50** avança com honras para a **Questão 7**, onde será submetida ao protocolo definitivo de **Validação Cruzada Estratificada 10-Fold** cobrindo todas as $13.543$ instâncias do banco de dados.
