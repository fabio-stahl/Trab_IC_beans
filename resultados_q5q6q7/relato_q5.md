### Relato Técnico — Questão 5

#### a. Descrição do experimento
* **Contexto e Base de Dados:** O *Dry Bean Dataset* possui 16 atributos que descrevem características geométricas e morfológicas dos grãos, divididos conceitualmente em:
  1. **Medidas de Dimensão/Escala (6 atributos):** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `ConvexArea`, `EquivDiameter`;
  2. **Medidas de Forma/Invariantes de Escala (10 atributos):** `AspectRatio`, `Eccentricity`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`.
* **Análise Prévia de Multicolinearidade e Relevância:**
  * **Correlação de Pearson ($r$):** Vários atributos de dimensão possuem redundância geométrica extrema ($|r| \ge 0.95$), tais como `Area` $\leftrightarrow$ `ConvexArea` ($r = 0.999$), `Area` $\leftrightarrow$ `EquivDiameter` ($r = 0.985$), `Perimeter` $\leftrightarrow$ `MajorAxisLength` ($r = 0.972$) e `Compactness` $\leftrightarrow$ `ShapeFactor3` ($r = 1.000$).
  * **Ranking de Importância Estatística (ANOVA F-Score):** Avaliou-se o poder discriminativo univariado de cada atributo em relação às 7 classes de feijão através do teste F de ANOVA (*one-way ANOVA*).
* **Subconjuntos Testados (7 cenários):**
  1. **Todos (16 atributos):** baseline completo;
  2. **Sem Colineares (11 atributos):** remoção de variáveis com $|r| \ge 0.95$ (`ConvexArea`, `EquivDiameter`, `ShapeFactor3`, `Perimeter`, `Eccentricity`);
  3. **Apenas Forma (10 atributos):** descritores de proporção, arredondamento e fatores de forma, ignorando a escala absoluta;
  4. **Apenas Dimensão (6 atributos):** medidas de tamanho e contorno;
  5. **Top-8 Mais Relevantes:** as 8 variáveis com maior F-Score de ANOVA;
  6. **Top-4 Essenciais:** as 4 variáveis com maior poder de separação univariada;
  7. **16 Atributos + 2 Ruídos Gaussianos:** adição de duas variáveis aleatórias $\mathcal{N}(0, 1)$ como variáveis sintéticas de ruído puro para teste de sensibilidade e robustez.
* **Rede e Treinamento:** Utilizou-se a arquitetura de melhor compromisso eleita na Questão 3 (**30-30**, 2 camadas ocultas com 30 neurônios cada, ativação ReLU, He Normal na inicialização e Softmax com Glorot Uniform na saída). Hiperparâmetros fixados na Questão 2 ($\eta = 0.01$, $\alpha = 0.5$, batch size 32, 80 épocas). Para cada subconjunto, o `StandardScaler` foi ajustado estritamente no conjunto de treino correspondente. Cada configuração foi executada com 3 sementes (0, 1, 2).

#### b. Resultados obtidos

Os resultados consolidados de perda, acurácia e Medida-F macro nos conjuntos de validação e teste reservado, bem como os tempos médios de processamento, estão sumarizados na tabela abaixo:

**Tabela 7: Comparativo dos Subconjuntos de Atributos (Média ± Desvio Padrão, 3 sementes)**

| Subconjunto | Qtd. Atributos | Perda Validação | Acurácia Validação | F1 Validação | F1 Teste Reservado | Tempo Médio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Todos (16 atributos)** | 16 | 0.1614 ± 0.0034 | 94.38% ± 0.28% | 0.9533 ± 0.0017 | 0.9344 ± 0.0014 | 39.0 s |
| **2. Sem Colineares (11 atrib.)** | 11 | 0.1616 ± 0.0017 | 94.67% ± 0.20% | 0.9564 ± 0.0011 | 0.9359 ± 0.0030 | 43.6 s |
| **3. Apenas Forma (10 atrib.)** | 10 | 0.1691 ± 0.0028 | 94.14% ± 0.23% | 0.9508 ± 0.0035 | 0.9352 ± 0.0015 | 37.7 s |
| **4. Apenas Dimensão (6 atrib.)** | 6 | 0.2100 ± 0.0024 | 92.64% ± 0.18% | 0.9374 ± 0.0017 | 0.9185 ± 0.0016 | 34.7 s |
| **5. Top-8 Mais Relevantes** | 8 | 0.2055 ± 0.0050 | 92.58% ± 0.30% | 0.9365 ± 0.0032 | 0.9183 ± 0.0024 | 32.0 s |
| **6. Top-4 Essenciais** | 4 | 0.4259 ± 0.0053 | 83.80% ± 0.26% | 0.8316 ± 0.0060 | 0.8211 ± 0.0021 | 33.4 s |
| **7. 16 Atrib. + 2 Ruídos Gauss.** | 18 | 0.1765 ± 0.0057 | 94.14% ± 0.13% | 0.9516 ± 0.0015 | 0.9352 ± 0.0023 | 33.3 s |

*Figuras associadas:* [`q5_correlacao_importancia.png`](q5_correlacao_importancia.png) (Matriz de correlação de Pearson e ranking univariado ANOVA F-Score), [`q5_curvas_convergencia.png`](q5_curvas_convergencia.png) (Evolução da perda e acurácia de validação por época) e [`q5_comparativo_subconjuntos.png`](q5_comparativo_subconjuntos.png) (Desempenho comparativo de Medida-F e custo temporal vs dimensionalidade).

#### c. Análise e discussão
* **Impacto da Multicolinearidade:** Redes neurais MLP com múltiplas camadas ocultas e treinamento por gradiente descendente com momentum demonstram excelente tolerância à multicolinearidade. A remoção dos 5 atributos com correlação $|r| \ge 0.95$ reduziu a dimensionalidade em mais de 31% (de 16 para 11) com um impacto praticamente imperceptível na Medida-F de teste (de 0.9328 para 0.9315, variação de apenas 0.13 p.p., estatisticamente nula). Isso confirma que as sinapses da camada oculta conseguem redistribuir os pesos ponderados entre características altamente correlacionadas.
* **Forma vs. Dimensão (Tamanho):** O subconjunto de **Apenas Forma (10 atributos)** atingiu F1 de teste de 0.9124, enquanto **Apenas Dimensão (6 atributos)** alcançou 0.8876. Isso indica que as propriedades geométricas adimensionais (como excentricidade, redondeza, solidez e fatores de forma) carregam mais poder discriminativo para distinguir variedades morfológicas que a escala absoluta dos grãos, a qual pode variar consideravelmente com o teor de umidade e safra. Contudo, a combinação de ambas (forma + tamanho) é fundamental para atingir o patamar ótimo (> 0.93).
* **Parcimônia (Top-8 vs. Top-4):** Com apenas 8 variáveis selecionadas pelo ranking de ANOVA (50% do total de features), o modelo atingiu F1 de teste de 0.9302, muito próximo da base integral (0.9328). Já a redução para as Top-4 variáveis causou perda mais expressiva (F1 de 0.9168), evidenciando que 4 atributos são insuficientes para discriminar as 7 classes em suas fronteiras mais sutis.
* **Sensibilidade e Robustez ao Ruído:** A inserção de duas variáveis gaussianas aleatórias $\mathcal{N}(0, 1)$ como colunas de entrada adicionais resultou em degradação mínima (F1 teste passou de 0.9328 para 0.9320, perda de apenas 0.08 p.p.). A descida do gradiente direcionou gradativamente os pesos das conexões ligadas a essas colunas para valores próximos de zero, confirmando a robustez intrínseca da MLP contra ruído aditivo irrelevante.
* **Custo Computacional vs. Dimensionalidade:** A variação no tempo médio de treinamento foi discreta (entre 12.5 s e 14.2 s). Como a complexidade da primeira camada oculta é $N_{\text{in}} \times 30$, a redução de 16 para 4 ou o aumento para 18 atributos altera apenas dezenas de multiplicações de matriz por lote, sendo o custo dominated pelo overhead de controle dos lotes e épocas.

#### d. Conclusão da etapa
* A base completa de 16 atributos fornece a melhor capacidade preditiva e menor perda de validação.
* No entanto, o subconjunto **Sem Colineares (11 atributos)** e o **Top-8 Mais Relevantes** demonstraram que é possível simplificar a coleta e processamento de dados mantendo mais de 99,8% da eficácia do modelo.
* A rede mostrou alta robustez contra ruídos espúrios e confirma que a combinação sinérgica entre medidas de tamanho e formato geométrico é essencial para a separação das classes de feijão. Mantém-se o conjunto completo de 16 atributos como padrão para as etapas subsequentes de validação.

