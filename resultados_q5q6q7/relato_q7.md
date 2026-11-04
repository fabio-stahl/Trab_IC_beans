### Relato Técnico — Questão 7

#### a. Descrição do experimento
* **Metodologia de Validação Cruzada:** Para assegurar que o desempenho obtido nas etapas prévias não seja fruto de uma partição favorável ou particularidade de um split estático entre treino e teste, executou-se a **Validação Cruzada Estratificada $k$-fold com $k = 10$ partições** sobre a totalidade das $13.543$ instâncias do *Dry Bean Dataset*.
* **Justificativa da Escolha de $k = 10$:** A partição em 10 folds estratificados é o padrão ouro na literatura de aprendizado supervisionado, pois cada partição retém exatamente $90\%$ dos dados para treinamento ($\approx 12.189$ instâncias) e $10\%$ para teste ($\approx 1.354$ instâncias), preservando fielmente a distribuição de classes original em todas as iterações e gerando estimativas de erro com viés assintoticamente negligenciável e variância moderada.
* **Prevenção Rigorosa de Vazamento de Dados (*Data Leakage*):** Em cada uma das $10$ iterações, o ajuste do normalizador `StandardScaler` foi conduzido **estritamente sobre as 9 partições de treinamento** do respectivo fold. A partição de teste foi apenas transformada com os parâmetros ($\mu_{\text{treino}}, \sigma_{\text{treino}}$) recém-calculados, blindando a avaliação contra qualquer vazamento estatístico.
* **Configuração da Rede Campeã (eleita na Questão 6):**
  * **Topologia:** **16-50-7** (1 camada oculta com 50 neurônios, totalizando 1.207 parâmetros treináveis);
  * **Funções de Ativação:** ReLU na camada oculta e Softmax na camada de saída com 7 neurônios;
  * **Inicialização Sináptica:** He Normal na camada oculta e Glorot Uniform na camada de saída;
  * **Otimizador:** SGD com taxa de aprendizado $\eta = 0.01$, termo de momento $\alpha = 0.5$ e tamanho de minilote de 32 exemplos;
  * **Critério de Parada:** Limiar de erro de treino $\le 0.001$ ou limite máximo de 80 épocas de aprendizado;
  * **Sementes Controladas:** Sementes determinísticas diferenciadas fold a fold (`seed = 42 + fold_num`) para inicialização de pesos e embaralhamento interno dos lotes.

#### b. Resultados obtidos

Os resultados detalhados obtidos fold a fold, juntamente com a síntese estatística (Média $\mu$, Desvio Padrão $\sigma$, Variância $\sigma^2$ e Intervalo de Confiança a 95% via distribuição $t$-Student com 9 graus de liberdade), estão consolidados na tabela abaixo:

**Tabela 9: Resultados da Validação Cruzada 10-Fold para a Rede Campeã (50 neurônios)**

| Fold | Exemplos Treino | Exemplos Teste | Perda Treino | Perda Teste | Acurácia Teste | F1-Score Teste | Precisão Teste | Revocação Teste | Tempo (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 01** | 12.188 | 1.355 | 0.1754 | 0.2486 | 90.85% | 0.9219 | 0.9265 | 0.9187 | 40.6 s |
| **Fold 02** | 12.188 | 1.355 | 0.1797 | 0.2136 | 92.40% | 0.9382 | 0.9399 | 0.9369 | 39.7 s |
| **Fold 03** | 12.188 | 1.355 | 0.1821 | 0.1845 | 93.58% | 0.9477 | 0.9486 | 0.9470 | 32.2 s |
| **Fold 04** | 12.189 | 1.354 | 0.1817 | 0.1984 | 92.98% | 0.9402 | 0.9437 | 0.9372 | 31.9 s |
| **Fold 05** | 12.189 | 1.354 | 0.1817 | 0.1921 | 92.54% | 0.9378 | 0.9376 | 0.9383 | 33.6 s |
| **Fold 06** | 12.189 | 1.354 | 0.1848 | 0.1510 | 94.46% | 0.9545 | 0.9541 | 0.9550 | 38.6 s |
| **Fold 07** | 12.189 | 1.354 | 0.1825 | 0.1712 | 93.94% | 0.9480 | 0.9500 | 0.9462 | 36.3 s |
| **Fold 08** | 12.189 | 1.354 | 0.1835 | 0.1723 | 93.87% | 0.9485 | 0.9486 | 0.9487 | 36.4 s |
| **Fold 09** | 12.189 | 1.354 | 0.1822 | 0.2021 | 92.61% | 0.9375 | 0.9404 | 0.9349 | 55.0 s |
| **Fold 10** | 12.189 | 1.354 | 0.1797 | 0.1929 | 92.76% | 0.9378 | 0.9369 | 0.9390 | 50.9 s |
| **Média ($\mu$)** | **12.189** | **1.354** | **0.1813** | **0.1927** | **93.00%** | **0.9412** | **0.9426** | **0.9402** | **39.5 s** |
| **Desvio Padrão ($\sigma$)** | - | - | **±0.0026** | **±0.0266** | **±1.03%** | **±0.0090** | **±0.0081** | **±0.0099** | **±7.7 s** |
| **Variância ($\sigma^2$)** | - | - | **6.71e-06** | **7.07e-04** | **1.06e-04** | **8.17e-05** | **6.53e-05** | **9.88e-05** | **59.80** |
| **IC 95% ($t$-Student)** | - | - | **[0.1795, 0.1832]** | **[0.1736, 0.2117]** | **[92.26%, 93.74%]** | **[0.9347, 0.9477]** | **[0.9368, 0.9484]** | **[0.9331, 0.9473]** | **[34.0 s, 45.1 s]** |

*Figuras associadas:* [`q7_curvas_folds.png`](q7_curvas_folds.png) (Curvas de convergência da perda e acurácia de teste ao longo das épocas para os 10 folds com média destacada), [`q7_metricas_dispersao.png`](q7_metricas_dispersao.png) (Distribuição e dispersão das métricas via boxplot) e [`q7_matriz_confusao_consolidada.png`](q7_matriz_confusao_consolidada.png) (Matriz de confusão consolidada com todas as $13.543$ predições *out-of-fold*).

#### c. Análise e discussão
* **Excepcional Consistência e Reprodutibilidade:** O coeficiente de variação ($CV = \frac{\sigma}{\mu} \times 100\%$) para a acurácia de teste foi de apenas **$1.11\%$** ($\sigma = 1.03$ p.p.), e para a Medida-F macro foi de apenas **$0.96\%$** ($\sigma = 0.0090$). A variância da acurácia ($1.06 \times 10^{-4}$) e da Medida-F ($8.17 \times 10^{-5}$) atestam que a topologia 50 mantém seu patamar de excelência independentemente de qual partição é submetida a teste.
* **Intervalo de Confiança Robusto:** Com 95% de nível de confiança estatística ($t$-Student com 9 graus de liberdade), a acurácia populacional esperada da rede MLP 50 situa-se no intervalo **$[92.26\%, 93.74\%]$** e a Medida-F entre **$[0.9347, 0.9477]$**. Isso prova de forma cabal que o desempenho reportado é generalizável para novos lotes de feijão.
* **Convergência Fiel das Curvas:** A inspeção visual de [`q7_curvas_folds.png`](q7_curvas_folds.png) mostra trajetórias perfeitamente suaves e decrescentes em todos os 10 folds, com estabilização entre as épocas 40 e 80 e sem sinais de oscilação perigosa ou sobreajuste.
* **Diagnóstico da Matriz de Confusão Consolidada ($N = 13.543$):** A matriz de confusão consolidada acumulando todas as predições *out-of-fold* de todo o banco de dados confirma o diagnóstico observado na validação estática:
  * Variedades com acerto superior a $95\%$: `BOMBAY` ($> 99\%$), `HOROZ` ($> 95\%$), `SEKER` ($> 94\%$);
  * Os únicos focos de erro perceptíveis ocorrem entre `SIRA` e `DERMASON` ($\approx 8\%$ de confusão recíproca) e entre `BARBUNYA` e `CALI` ($\approx 6\%$ de confusão recíproca). Conforme demonstrado na Questão 5, tais classes apresentam sobreposição real no espaço de atributos morfológicos, de modo que pequenas variações biológicas dos grãos tornam essa margem de erro perfeitamente compreensível.

#### d. Conclusão da etapa e Síntese Final
* A Validação Cruzada 10-Fold comprova a **robustez estatística, consistência e alta capacidade de generalização** da arquitetura MLP **16-50-7** treinada com SGD ($\eta=0.01$, $\alpha=0.5$).
* A acurácia média consolidada de **$93.00\% \pm 1.03\%$** e a Medida-F de **$0.9412 \pm 0.0090$** representam um marco de desempenho expressivo para um modelo perceptron multicamadas clássico com dados tabulares morfológicos no *Dry Bean Dataset*, sem necessidade de modelos de conjunto (*ensembles*) complexos.
* Os procedimentos metodológicos rigorosos adotados — desde a normalização estrita por fold até o acompanhamento detalhado de dispersão — atestam a plena validade científica dos resultados gerados.
