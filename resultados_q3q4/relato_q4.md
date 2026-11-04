### Relato Técnico — Questão 4

#### a. Descrição do experimento
* **Modelo:** melhor topologia da Q3, **16-30-30-7**: ReLU nas ocultas, Softmax na saída, Entropia Cruzada Categórica, SGD com $\eta = 0.01$, $\alpha = 0.5$ e lote 32. Tudo igual às etapas anteriores.
* **Tamanhos testados:** 5%, 10%, 20%, 40%, 60%, 80% e 100% do conjunto de treinamento, ou seja, de 460 a 9.208 exemplos. Incluímos 5% e 10% além dos tamanhos sugeridos (20–100%) para ver a região onde faltam dados.
* **Amostragem:** **estratificada** (`train_test_split(..., stratify=y)`), que preserva a proporção das 7 classes. Cada tamanho foi repetido com 3 sementes; a semente muda a amostra e os pesos iniciais.
* **Qualidade dos exemplos:** (i) com 20% dos dados, amostra estratificada × aleatória simples × **enviesada**, em que a BARBUNYA cai de 9,8% para 1,1% da amostra; (ii) **ruído de rótulos**, com 10%, 20% e 30% dos exemplos de treino trocados para outra classe sorteada. Validação e teste sempre com rótulos corretos.
* **Parada:** perda de treino $\le 0.001$, ou 20 épocas sem melhora na perda de validação (máx. 300 épocas), restaurando os melhores pesos. Assim cada tamanho treina até convergir, e o número de épocas vira uma medida de **tempo de convergência**. O `StandardScaler` é ajustado só na amostra de treino usada.
* **Conjunto de teste:** 2.709 exemplos, usados apenas para medir a generalização, como pede o enunciado. Nenhuma decisão foi tomada com base nele.

#### b. Resultados obtidos

**Tabela 5: quantidade de exemplos (média ± desvio, 3 sementes)**

| Fração | Exemplos | Época de convergência | Tempo (s) | Perda treino | Perda val | Perda teste | Acurácia val | Acurácia teste | F1 val | F1 teste |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5% | 460 | 161 ± 32 | 18.3 ± 3.9 | 0.1360 ± 0.0108 | 0.2299 ± 0.0141 | 0.2802 ± 0.0129 | 92.15% ± 0.60 | 90.65% ± 0.58 | 0.9335 ± 0.0012 | 0.9187 ± 0.0049 |
| 10% | 920 | 144 ± 14 | 16.4 ± 1.3 | 0.1537 ± 0.0186 | 0.1941 ± 0.0038 | 0.2558 ± 0.0144 | 93.28% ± 0.28 | 91.33% ± 0.15 | 0.9426 ± 0.0016 | 0.9244 ± 0.0019 |
| 20% | 1.841 | 119 ± 12 | 15.2 ± 1.3 | 0.1603 ± 0.0086 | 0.1794 ± 0.0025 | 0.2393 ± 0.0141 | 93.99% ± 0.25 | 91.61% ± 0.43 | 0.9488 ± 0.0015 | 0.9266 ± 0.0040 |
| 40% | 3.683 | 105 ± 43 | 15.6 ± 5.1 | 0.1706 ± 0.0032 | 0.1715 ± 0.0026 | 0.2200 ± 0.0058 | 94.22% ± 0.21 | 92.01% ± 0.11 | 0.9513 ± 0.0017 | 0.9303 ± 0.0017 |
| 60% | 5.524 | 95 ± 35 | 16.2 ± 4.8 | 0.1720 ± 0.0047 | 0.1639 ± 0.0011 | 0.2158 ± 0.0064 | 94.44% ± 0.09 | 92.27% ± 0.29 | 0.9531 ± 0.0002 | 0.9327 ± 0.0025 |
| 80% | 7.366 | 67 ± 19 | 13.8 ± 2.8 | 0.1744 ± 0.0039 | 0.1642 ± 0.0005 | 0.2097 ± 0.0040 | 94.22% ± 0.31 | 92.49% ± 0.17 | 0.9518 ± 0.0025 | 0.9346 ± 0.0018 |
| 100% | 9.208 | 79 ± 20 | 16.2 ± 2.5 | 0.1725 ± 0.0061 | 0.1619 ± 0.0015 | 0.2073 ± 0.0044 | 94.38% ± 0.20 | 92.57% ± 0.28 | 0.9534 ± 0.0009 | 0.9357 ± 0.0024 |

**Tabela 6: qualidade e representatividade (média ± desvio, 3 sementes)**

| Cenário | Exemplos | Épocas | Acurácia treino | Perda val | F1 val | F1 teste | Revocação BARBUNYA (teste) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 20% estratificada | 1.841 | 139 ± 12 | 94.08% | 0.1794 | 0.9488 ± 0.0015 | 0.9266 ± 0.0040 | 0.897 ± 0.002 |
| 20% aleatória simples | 1.841 | 132 ± 31 | 94.02% | 0.1849 | 0.9481 ± 0.0030 | 0.9282 ± 0.0041 | 0.897 ± 0.017 |
| 20% enviesada (BARBUNYA 9,8% → 1,1%) | 1.679 | 177 ± 21 | 94.66% | 0.2026 | 0.9388 ± 0.0014 | 0.9197 ± 0.0044 | **0.782 ± 0.006** |
| 100%, 0% de rótulos trocados | 9.208 | 99 ± 20 | 93.56% | 0.1619 | 0.9534 ± 0.0009 | 0.9357 ± 0.0024 | 0.911 |
| 100%, 10% de rótulos trocados | 9.208 | 170 ± 57 | 84.34% | 0.2575 | 0.9515 ± 0.0024 | 0.9341 ± 0.0009 | 0.913 |
| 100%, 20% de rótulos trocados | 9.208 | 232 ± 85 | 75.20% | 0.3699 | 0.9497 ± 0.0044 | 0.9311 ± 0.0006 | 0.913 |
| 100%, 30% de rótulos trocados | 9.208 | 224 ± 60 | 65.69% | 0.4826 | 0.9530 ± 0.0033 | 0.9305 ± 0.0041 | 0.898 |

Figuras: [`q4_curvas_generalizacao.png`](q4_curvas_generalizacao.png) (curvas de generalização no treino, validação e teste), [`q4_curvas_epocas.png`](q4_curvas_epocas.png) e [`q4_qualidade.png`](q4_qualidade.png).

#### c. Análise e discussão
* **Curvas de generalização: melhora contínua com saturação.** A Medida-F no teste sobe de forma **monotônica** com o tamanho do treino: 0.9187 → 0.9244 → 0.9266 → 0.9303 → 0.9327 → 0.9346 → 0.9357. O ganho por exemplo, porém, cai rápido. Dobrar de 5% para 10% rende +0.57 p.p., e de 20% para 100% (5× mais dados) rende só +0.91 p.p. Na validação a curva **satura por volta de 40–60%** (F1 0.951–0.953); acima disso, as diferenças ficam dentro do desvio entre sementes. A perda de validação segue o mesmo padrão (0.230 → 0.172 em 40% → 0.162 em 100%).
* **Overfitting com poucos dados:** com 460 exemplos a rede quase memoriza a amostra. A perda de treino é a menor (0.136), mas a de teste é a maior (0.280), um gap de 0.144. Conforme a amostra cresce, a perda de treino **sobe** (fica mais difícil ajustar todos os exemplos) e a de validação e teste **cai**, e as curvas se aproximam: com 100% o gap treino–teste é de 0.035. Mais dados funcionam aqui como regularização.
* **Validação × teste:** o teste fica sistematicamente ≈ 1,5–2 p.p. abaixo da validação em F1. Há duas causas: a validação foi usada para a parada antecipada (e, nas etapas anteriores, para escolher os hiperparâmetros), o que a deixa levemente otimista; e há variação natural entre amostras. A tendência é a mesma nos dois conjuntos, então as conclusões se mantêm. Essa diferença reforça a importância de manter um conjunto de teste reservado.
* **Representatividade:** com 1.841 exemplos, a amostra **aleatória simples** já reproduz bem as proporções das classes (BARBUNYA 9,4% contra 9,8%). Por isso teve o mesmo desempenho médio da estratificada (F1 teste 0.9282 × 0.9266). A estratificação reduz a **variância**: o desvio da revocação de BARBUNYA cai de 0.017 para 0.002, e o ganho seria maior com amostras pequenas ou classes raras. Já a amostra **enviesada** mostra o efeito de uma base não representativa. Com apenas 18 exemplos de BARBUNYA, a revocação dessa classe no teste cai de 0.897 para **0.782 (−11,5 p.p.)**: a rede passa a atribuir esses grãos às classes vizinhas. O F1 macro cai só 0.7 p.p. e a acurácia de treino até sobe, porque a métrica global esconde o problema. Uma classe sub-representada é prejudicada mesmo quando o modelo "parece" bom.
* **Ruído de rótulos (qualidade):** a acurácia de treino, medida contra os rótulos ruidosos, cai quase exatamente na proporção do ruído (93.6% → 65.7% com 30%). Isso indica que a rede **não memorizou os rótulos errados**. A acurácia e o F1 no teste caem só 0.5 p.p. (0.9357 → 0.9305). Essa robustez vem do ruído ser uniforme (não favorece nenhuma classe), da capacidade pequena da 30-30 e da parada antecipada guiada por uma validação limpa. O preço aparece em duas frentes. A **perda de validação triplica** (0.162 → 0.483): as probabilidades ficam bem menos confiantes e calibradas. E o treino fica **mais lento**: a convergência passa de ~99 para ~230 épocas e o tempo dobra (16 s → 38 s).
* **Tempo de convergência:** não cresce na proporção dos dados. Cada época custa mais com mais exemplos (0.10 s com 5% contra 0.17 s com 100%, só 1,7× para 20× mais dados, por causa do overhead fixo por época). Em compensação, com mais dados a rede converge em **menos épocas** (161 → 79), porque cada época faz mais atualizações de pesos. O tempo total fica em torno de 14–18 s em todos os tamanhos. Nesta base, o custo computacional não limita o uso de 100% dos dados.

#### d. Conclusão da etapa
* **Quantidade:** mais dados sempre ajudaram, com retornos decrescentes. Com **≈ 40–60% do treino (3.700–5.500 exemplos)** o modelo já alcança ≈ 99,5–99,8% da Medida-F final (validação e teste). Com menos de 20% o ganho por exemplo ainda é alto e há overfitting visível.
* **Qualidade:** a **representatividade das classes** pesa mais que o volume. Sub-representar uma classe custa mais de 11 p.p. de revocação nela e passa quase despercebido nas métricas globais. O ruído uniforme de rótulos afeta pouco a acurácia, mas degrada a calibração e dobra o tempo de treino. A amostragem estratificada é a escolha segura, principalmente para amostras pequenas.
* **Ponto de equilíbrio:** como o tempo de treino praticamente não depende do tamanho da base, recomenda-se usar **100% dos dados de treino**, que deu a melhor generalização no teste (F1 0.9357). Se coletar e rotular dados fosse caro, ~40–60% (estratificado) seria o mínimo razoável.
* Na etapa de Validação Inicial, as quatro topologias selecionadas na Q3 serão comparadas no conjunto de teste, treinadas com o conjunto de treinamento completo.
