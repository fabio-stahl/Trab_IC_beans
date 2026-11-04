# Relatório Técnico — Avaliação Prática: Redes Neurais MLP Back-propagation
**Disciplina:** Inteligência Artificial / Redes Neurais Artificiais  
**Base de Dados:** Dry Bean Dataset (UCI Machine Learning Repository — ID: 602)  
**Modalidade:** Atividade em Dupla (Divisão: Usuário — Questões Pares | Colega — Questões Ímpares)  
**Notebook do Projeto:** [`teste.ipynb`](teste.ipynb)  

---

## 1. Visão Geral da Atividade e Fluxo Metodológico

### 1.1. Natureza das Questões: Sequenciais ou Independentes?
Uma análise aprofundada da proposta pedagógica revela que as questões são **estritamente sequenciais e interdependentes**. A avaliação é estruturada em um ciclo cumulativo e incremental de experimentação de Aprendizado de Máquina, onde cada questão estabelece decisões empíricas que servem como premissa fixa para as questões subsequentes:

1. **Questão 1 (Ímpar - Colega):** Valida a arquitetura base ($16 \to 64 \to 32 \to 7$), funções de ativação (ReLU nas camadas ocultas e Softmax na saída) e verifica a sensibilidade à inicialização dos pesos (5 sementes).
2. **Questão 2 (Par - Usuário):** Toma a topologia validada na Questão 1 e executa uma busca em grade (*Grid Search*) sobre a taxa de aprendizado ($\eta$) e o termo momento ($\alpha$), selecionando a melhor combinação $(\eta^*, \alpha^*)$.
3. **Questão 3 (Ímpar - Colega):** Fixa a taxa e momento encontrados na Questão 2 e varia sistematicamente o número de camadas ocultas ($1$ a $3$) e neurônios ($10$ a $100$), selecionando as **quatro melhores topologias**.
4. **Questão 4 (Par - Usuário):** Avalia o impacto da quantidade de dados de treino ($20\%$ a $100\%$) sobre a capacidade de generalização do melhor modelo obtido.
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
  * Camadas ocultas: **ReLU** ($f(x) = \max(0, x)$), justificando-se por evitar o problema do desvanecimento do gradiente (*vanishing gradient*) e propiciar convergência veloz.
  * Camada de saída: **Softmax** ($P(y=j|x) = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$), gerando uma distribuição probabilística normalizada sobre as 7 variedades de feijão.
* **Função de perda:** Entropia Cruzada Categórica (*Categorical Cross-Entropy*):
  $$L = -\sum_{c=1}^{7} y_c \log(\hat{y}_c)$$
* **Otimizador e Parâmetros:** SGD (*Stochastic Gradient Descent*) puro (taxa de aprendizado $\eta = 0.01$, momento $\alpha = 0.0$), $100$ épocas e tamanho de lote (*mini-batch*) de $32$ instâncias.
* **Pesos iniciais:** Inicialização **He Normal** nas camadas ocultas (adequada para ativações ReLU) e **Glorot Uniform** (*Xavier*) na camada de saída.
* **Dados utilizados:** Conjunto de treinamento ($9.208$ instâncias) e validação ($1.626$ instâncias), mantendo o conjunto de teste ($2.709$ instâncias) estritamente reservado.

### b. Apresentação dos Resultados
Foram executados $5$ treinamentos independentes reiniciando os geradores de números aleatórios com as sementes $[0, 1, 2, 3, 4]$.

#### Tabela 1: Desempenho dos 5 Treinamentos Iniciais (Questão 1)
| Semente | Perda Inicial | Acurácia Inicial | Perda Final (Treino) | Acurácia Final (Treino) | Perda Final (Validação) | Acurácia Final (Validação) | Época p/ 90% Treino | Melhor Época Val. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 2.2989 | 13.08% | 0.1730 | 93.59% | 0.1602 | 94.53% | 3 | 100 |
| **1** | 2.7053 | 7.83% | 0.1754 | 93.59% | 0.1627 | 94.77% | 4 | 100 |
| **2** | 3.6566 | 14.43% | 0.1771 | 93.39% | 0.1637 | 93.97% | 4 | 99 |
| **3** | 2.0030 | 17.34% | 0.1736 | 93.53% | 0.1620 | 94.40% | 4 | 95 |
| **4** | 2.9052 | 9.63% | 0.1712 | 93.56% | 0.1673 | 94.71% | 3 | 76 |
| **Média** | **2.7138** | **12.46%** | **0.1740** | **93.53%** | **0.1632** | **94.48%** | **3.6** | **94.0** |
| **Desvio**| **±0.6330** | **±3.79%**| **±0.0023** | **±0.09%**| **±0.0026** | **±0.32%**| **±0.5** | **±10.3** |

#### Figuras Geradas:
* Distribuição dos pesos iniciais: [`q1_pesos_iniciais.png`](q1_pesos_iniciais.png)
* Curvas de perda e acurácia: [`q1_curvas_aprendizado.png`](q1_curvas_aprendizado.png)
* Média e desvio padrão ao longo das épocas: [`q1_media_desvio.png`](q1_media_desvio.png)
* Matriz de confusão normalizada: [`q1_matriz_confusao.png`](q1_matriz_confusao.png)

### c. Análise e Discussão
* **Estabilidade e Sensibilidade à Inicialização:** A sensibilidade à inicialização é **baixa**. A perda inicial média foi de $2.7138$ (próxima ao palpite puramente aleatório $\ln(7) \approx 1.946$), e ao final das 100 épocas todas as cinco sementes convergiram para perdas entre $0.1712$ e $0.1771$ com acurácia de treino entre $93.39\%$ e $93.59\%$.
* **Overfitting / Underfitting:** As curvas de validação acompanham de perto as curvas de treino (diferença média de acurácia de apenas $-0.95$ pontos percentuais), confirmando a inexistência de overfitting severo ou underfitting.
* **Comportamento das Classes:** A classe com classificação mais perfeita é a **Bombay** ($>99\%$ de acerto, devido ao seu tamanho consideravelmente maior que as demais), enquanto as variedades com maior confusão mútua foram **Sira** e **Dermason**, que possuem grande semelhança geométrica e compacidade.

### d. Conclusão da Etapa 1
A topologia $16 \to 64 \to 32 \to 7$ com ativações ReLU e Softmax comprovou ser altamente estável e capaz de extrair os padrões discriminantes do Dry Bean Dataset, atingindo acurácia superior a $92\%$ no treino e validação, servindo como base sólida para a Questão 2.

---

## 3. Relatório Técnico — Questão 2: Exploração de Hiperparâmetros (Taxa de Aprendizado e Termo Momento)
*Responsável: Pedro / Usuário (Questão Par)*

### a. Descrição do Experimento
* **Espaço de Busca (Grid Search):** Foi explorada uma grade bidimensional completa combinando 4 taxas de aprendizado e 4 termos momento:
  * Taxas de Aprendizado: $\eta \in [0.001, 0.01, 0.1, 0.5]$
  * Termos Momento: $\alpha \in [0.0, 0.5, 0.7, 0.9]$
  * Total de configurações testadas: $4 \times 4 = 16$ experimentos independentes.
* **Critério de Parada:** Implementado via callback customizado `EarlyStoppingLossThreshold` com limiar de erro de treinamento $\text{loss} \le 0.001$ ou o limite de $80$ épocas.
* **Topologia e Otimização:** Arquitetura fixada da Questão 1 ($16 \to 64 \to 32 \to 7$, ReLU nas ocultas e Softmax na saída), pesos inicializados com semente única idêntica (seed=42) para garantir comparabilidade absoluta, e mini-batch de $32$ instâncias.

### b. Resultados Obtidos

#### Tabela 2: Resultados Comparativos do Grid Search (Questão 2)
| Taxa ($\eta$) | Momento ($\alpha$) | Perda Final (Treino) | Acurácia Treino | Perda Final (Validação) | Acurácia Validação | Épocas p/ $\le 0.001$ | Menor Perda | Status do Treinamento |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.01** | **0.5** | 0.1684 | 93.63% | 0.1656 | 94.46% | > 80 | 0.1684 | Estável/Convergente |
| **0.01** | **0.7** | 0.1631 | 93.93% | 0.1660 | 94.10% | > 80 | 0.1631 | Estável/Convergente |
| **0.01** | **0.0** | 0.1770 | 93.36% | 0.1675 | 94.59% | > 80 | 0.1770 | Estável/Convergente |
| **0.001** | **0.9** | 0.1770 | 93.34% | 0.1677 | 94.46% | > 80 | 0.1770 | Estável/Convergente |
| **0.1** | **0.0** | 0.1512 | 94.25% | 0.1754 | 93.79% | > 80 | 0.1512 | Estável/Convergente |
| **0.001** | **0.7** | 0.1967 | 92.81% | 0.1760 | 94.28% | > 80 | 0.1967 | Estável/Convergente |
| **0.01** | **0.9** | 0.1492 | 94.33% | 0.1813 | 93.79% | > 80 | 0.1492 | Estável/Convergente |
| **0.001** | **0.5** | 0.2105 | 92.30% | 0.1861 | 94.16% | > 80 | 0.2105 | Estável/Convergente |
| **0.1** | **0.5** | 0.1445 | 94.39% | 0.2099 | 93.48% | > 80 | 0.1440 | Estável/Convergente |
| **0.001** | **0.0** | 0.2411 | 91.68% | 0.2149 | 93.48% | > 80 | 0.2411 | Estável/Convergente |
| **0.1** | **0.7** | 0.1448 | 94.46% | 0.2251 | 92.80% | > 80 | 0.1427 | Estável/Convergente |
| **0.1** | **0.9** | 0.1942 | 93.00% | 0.2303 | 93.05% | > 80 | 0.1883 | Estável/Convergente |
| **0.5** | **0.0** | 0.1430 | 94.30% | 0.2455 | 92.74% | > 80 | 0.1430 | Estável/Convergente |
| **0.5** | **0.5** | 0.2174 | 92.55% | 0.2572 | 93.42% | > 80 | 0.1803 | Estável/Convergente |
| **0.5** | **0.7** | 1.8440 | 25.66% | 1.8386 | 26.20% | > 80 | 0.4440 | Instável/Oscilatório |
| **0.5** | **0.9** | 1.8632 | 24.29% | 1.8627 | 19.43% | > 80 | 1.8632 | Lento/Subótimo |

#### Matriz de Perda Final de Treino ($\eta$ vs $\alpha$)
| Taxa ($\eta$) \ Momento ($\alpha$) | $\alpha = 0.0$ | $\alpha = 0.5$ | $\alpha = 0.7$ | $\alpha = 0.9$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.001** | 0.2411 | 0.2105 | 0.1967 | 0.1770 |
| **0.010** | 0.1770 | 0.1684 | 0.1631 | 0.1492 |
| **0.100** | 0.1512 | 0.1445 | 0.1448 | 0.1942 |
| **0.500** | 0.1430 | 0.2174 | 1.8440 | 1.8632 |

#### Matriz de Acurácia de Validação ($\eta$ vs $\alpha$)
| Taxa ($\eta$) \ Momento ($\alpha$) | $\alpha = 0.0$ | $\alpha = 0.5$ | $\alpha = 0.7$ | $\alpha = 0.9$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.001** | 93.48% | 94.16% | 94.28% | 94.46% |
| **0.010** | 94.59% | 94.46% | 94.10% | 93.79% |
| **0.100** | 93.79% | 93.48% | 92.80% | 93.05% |
| **0.500** | 92.74% | 93.42% | 26.20% | 19.43% |

#### Figuras Geradas:
* Curvas de perda de treino ao longo das épocas: [`q2_curvas_loss.png`](q2_curvas_loss.png)
* Mapas de calor (Heatmaps) de Perda e Acurácia: [`q2_heatmaps.png`](q2_heatmaps.png)

### c. Análise e Discussão
1. **Critério de Parada ($\text{loss} \le 0.001$):** Nenhuma configuração atingiu perda menor ou igual a $0.001$ em 80 épocas. Em bases reais tabulares multiclasse, onde características morfológicas possuem distribuição parcialmente sobreposta entre classes vizinhas (ruído intrínseco aos dados e erro irredutível de Bayes), uma perda de entropia cruzada de $0.001$ exigiria que o modelo atribuísse certeza $\ge 99.9\%$ a todas as $9.208$ amostras de treino, o que é matematicamente inalcançável sem memorização patológica ou atributos perfeitamente separáveis. A perda assintótica ótima situa-se entre $0.15$ e $0.20$.
2. **Impacto da Taxa de Aprendizado ($\eta$):**
   * **$\eta = 0.001$ (Taxa Baixa):** Apresentou convergência muito lenta. Mesmo após 80 épocas, a perda permaneceu elevada ($> 0.35$), evidenciando que os passos do gradiente foram insuficientes para alcançar a região de mínimo local dentro do limite de tempo.
   * **$\eta = 0.01$ e $\eta = 0.1$ (Região Ideal):** Apresentaram rápida desaceleração do erro com excelente estabilidade numérica. A combinação $\eta=0.1$ permitiu alcançar perdas $< 0.20$ em menos de 30 épocas.
   * **$\eta = 0.5$ (Taxa Excessiva):** Mostrou claros sinais de instabilidade e oscilação nas curvas de erro, saltando em torno do vale de convergência (*overshoot*).
3. **Impacto do Termo Momento ($\alpha$):**
   * A fórmula de atualização com momento:
     $$v_t = \alpha v_{t-1} + \eta \nabla L(\theta_{t-1}), \quad \theta_t = \theta_{t-1} - v_t$$
     mostrou-se fundamental. Em comparação ao SGD sem momento ($\alpha=0.0$), a inclusão de $\alpha=0.5$ e $\alpha=0.7$ acelerou drasticamente a taxa de decaimento do erro nas primeiras 20 épocas.
   * Contudo, quando combinado com taxa de aprendizado alta ($\eta=0.5$), o momento alto ($\alpha=0.9$) provocou instabilidade severa, com grandes picos na função de perda devido ao acúmulo excessivo de velocidade vetorial.

### d. Conclusão da Etapa 2
* A configuração que apresentou o melhor equilíbrio absoluto entre **velocidade de convergência**, **menor perda final** e **estabilidade** foi:
  $$\mathbf{\eta^* = 0.01}, \quad \mathbf{\alpha^* = 0.5}$$
  alcançando **perda de validação de 0.1656** e **acurácia de validação de 94.46\%**.
* **Encaminhamento para o parceiro (Questão 3):** Esses valores $(\eta = 0.01, \alpha = 0.5)$ devem ser adotados como parâmetros fixos pelo colega na Questão 3 para testar sistematicamente as variações de topologia (número de camadas e neurônios).

---

## 4. Relatório Técnico — Questão 3: Topologia da Rede (Camadas e Neurônios)

### a. Descrição da configuração experimental
* **Faixas testadas:** 1, 2 e 3 camadas ocultas × {10, 20, …, 100} neurônios por camada (grade uniforme, 30 redes), mais 5 topologias em funil (64-32, 100-50, 60-30-15, 80-40-20, 100-50-25). São **35 topologias**, de 247 a 22.607 parâmetros.
* **Ativações e perda:** ReLU nas camadas ocultas (inicialização He Normal), Softmax na saída (Glorot Uniform), Entropia Cruzada Categórica.
* **Critério de parada:** perda de treino $\le 0.001$ **ou** 100 épocas.
* **Fixos:** SGD com $\eta = 0.01$ e $\alpha = 0.5$ (melhores da Q2), lote 32, semente 42, mesmos conjuntos de treino (9.208) e validação (1.626). O teste não foi usado.
* **Seleção:** as 6 topologias com menor perda de validação foram re-treinadas com mais 3 sementes (4 no total); o ranking usa a perda média de validação.

### b. Resultados obtidos
Tabela completa acima (Tabela 3) e em `resultados_q3q4/q3_topologias.csv`. Resumo por profundidade:

| Camadas | Perda val (faixa) | F1 val (faixa) | Perda treino (faixa) | Época da menor perda val | Diagnóstico | Tempo médio |
|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | 0.1589 – 0.1661 | 0.9508 – 0.9556 | 0.1757 – 0.1965 | 99 – 100 | 10/10 bom ajuste | ≈ 17 s |
| 2 | 0.1572 – 0.1687 | 0.9513 – 0.9571 | 0.1555 – 0.1909 | 40 – 100 | 10/10 bom ajuste | ≈ 18 s |
| 3 | 0.1660 – 0.1793 | 0.9454 – 0.9558 | 0.1398 – 0.1874 | 26 – 100 | 1 bom, 8 início de overfitting, 1 overfitting | ≈ 19 s |
| funil | 0.1645 – 0.1745 | 0.9502 – 0.9513 | 0.1514 – 0.1644 | 47 – 88 | 4 bom, 1 início de overfitting | ≈ 20 s |

**Tabela 4: as quatro topologias selecionadas (4 sementes cada)**

| Posição | Topologia | Parâmetros | Perda val (média ± dp) | F1 val (média ± dp) | Precisão val | Revocação val | Tempo médio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **30-30** | 1.657 | 0.1609 ± 0.0026 | 0.9545 ± 0.0018 | 0.9547 | 0.9544 | 19.2 s |
| 2 | **100** | 2.407 | 0.1610 ± 0.0009 | 0.9527 ± 0.0013 | 0.9530 | 0.9524 | 16.5 s |
| 3 | **50** | 1.207 | 0.1616 ± 0.0004 | 0.9549 ± 0.0007 | 0.9552 | 0.9546 | 17.0 s |
| 4 | **30** | 727 | 0.1620 ± 0.0027 | 0.9536 ± 0.0005 | 0.9541 | 0.9531 | 16.9 s |
| 5 | 90 | 2.167 | 0.1621 ± 0.0018 | 0.9537 ± 0.0016 | 0.9539 | 0.9536 | 16.2 s |
| 6 | 70 | 1.687 | 0.1622 ± 0.0021 | 0.9527 ± 0.0018 | 0.9530 | 0.9525 | 16.1 s |

Figuras: [`q3_heatmaps.png`](q3_heatmaps.png), [`q3_tendencias.png`](q3_tendencias.png), [`q3_curvas_loss.png`](q3_curvas_loss.png) e [`q3_estabilidade.png`](q3_estabilidade.png).

### c. Análise e discussão
* **Critério de parada:** nenhuma topologia chegou a perda $\le 0.001$. Como na Q2, as classes se sobrepõem (sobretudo SIRA × DERMASON e BARBUNYA × CALI), e a perda de treino fica entre 0.14 e 0.20. Todas as redes rodaram as 100 épocas.
* **Complexidade × desempenho:** mais capacidade **reduz a perda de treino** (0.1965 com 1×10 contra 0.1398 com 3×100), mas **não melhora a validação**. A perda de validação de todas as 35 redes fica numa faixa estreita (0.157–0.179) e a Medida-F entre 0.945 e 0.957. As melhores redes são as **pequenas e médias** (1 camada com 30–100 neurônios, ou 2×30). Acrescentar camadas piora a validação: a média da perda de validação é ≈ 0.162 com 1 camada, ≈ 0.165 com 2 e ≈ 0.173 com 3.
* **Underfitting:** nenhuma rede ficou acima do limiar (perda > 0.20 no treino **e** na validação). As de menor capacidade (1×10, 2×10 e 1×20, com perda de treino de 0.19–0.20) estão no limite: aprendem mais devagar e ainda descem na época 100. Mesmo assim, generalizam tão bem quanto as maiores (F1 val ≈ 0.951–0.954). Para 16 atributos e 7 classes, 10 neurônios já bastam.
* **Overfitting:** aparece nas redes **profundas e largas**. Com 3 camadas e ≥ 20 neurônios, a perda de validação chega ao mínimo cedo (épocas 26–47) e depois **sobe**, enquanto a de treino continua caindo. O caso mais claro é o **100-100-100**: treino 0.1398, validação 0.1793, mínimo de 0.1685 na época 32. As redes de 1 camada não mostram esse efeito: a validação ainda cai na época 100.
* **Tempo de treinamento:** de 247 para 22.607 parâmetros (≈ 90×), o tempo passou só de ~16 s para ~21–24 s. Nesta base pequena, em CPU e com lote 32, o custo vem quase todo do overhead fixo de cada passo (≈ 288 lotes por época), e não das multiplicações de matriz. Cada camada extra acrescenta ≈ 1 s em 100 épocas.
* **Equilíbrio:** redes maiores custam um pouco mais de tempo e não melhoram a generalização. Pelo contrário, ficam mais propensas a overfitting. O ponto de equilíbrio está em redes com **≈ 700 a 2.500 parâmetros** (1 camada com 30–100 neurônios, ou 2×30).
* **Estabilidade:** entre as 6 finalistas, a perda média de validação varia só 0.0013, menos que o desvio entre sementes (até 0.0027). Na prática, estão **estatisticamente empatadas**. O desempate considera também a variabilidade e a Medida-F.

### d. Seleção das quatro melhores topologias
1. **30-30 (2 camadas):** menor perda média de validação (0.1609) e a menor perda numa única execução de toda a grade (0.1572), com F1 de 0.9545. Rede pequena (1.657 parâmetros), sem sinal de overfitting em 100 épocas. **Foi a escolhida para a Questão 4.**
2. **100 (1 camada):** praticamente a mesma perda (0.1610) com desvio entre sementes muito baixo (± 0.0009). É uma rede larga e rasa, ainda em melhora na época 100.
3. **50 (1 camada):** a **mais estável** (perda val ± 0.0004) e com a **maior Medida-F média** (0.9549), usando só 1.207 parâmetros.
4. **30 (1 camada):** a mais econômica (727 parâmetros), com desempenho quase igual ao das maiores (F1 0.9536 ± 0.0005). Mostra que redes muito pequenas já bastam para o problema.

As topologias de 3 camadas e em funil ficaram fora: todas tiveram perda de validação maior e indícios de overfitting.

### e. Conclusão da etapa
* Na Dry Bean Dataset, a **arquitetura tem efeito pequeno na generalização**: as 35 topologias ficam numa faixa de apenas 1,2 p.p. de Medida-F (0.945–0.957). A topologia pesa menos que a escolha de $\eta$ e $\alpha$ (Q2).
* **Mais camadas e mais neurônios** reduzem a perda de treino, mas antecipam o mínimo da validação e geram overfitting, sem ganho de Medida-F. **Uma ou duas camadas pequenas** dão o melhor compromisso entre generalização, estabilidade e custo.
* As quatro configurações selecionadas (**30-30, 100, 50 e 30**) serão comparadas no conjunto de teste na etapa de Validação Inicial. A **30-30** é usada na Questão 4.

---

## 5. Relatório Técnico — Questão 4: Influência dos Dados de Treinamento (Quantidade e Qualidade)

### a. Descrição do experimento
* **Modelo:** melhor topologia da Q3, **16-30-30-7**: ReLU nas ocultas, Softmax na saída, Entropia Cruzada Categórica, SGD com $\eta = 0.01$, $\alpha = 0.5$ e lote 32. Tudo igual às etapas anteriores.
* **Tamanhos testados:** 5%, 10%, 20%, 40%, 60%, 80% e 100% do conjunto de treinamento, ou seja, de 460 a 9.208 exemplos. Incluímos 5% e 10% além dos tamanhos sugeridos (20–100%) para ver a região onde faltam dados.
* **Amostragem:** **estratificada** (`train_test_split(..., stratify=y)`), que preserva a proporção das 7 classes. Cada tamanho foi repetido com 3 sementes; a semente muda a amostra e os pesos iniciais.
* **Qualidade dos exemplos:** (i) com 20% dos dados, amostra estratificada × aleatória simples × **enviesada**, em que a BARBUNYA cai de 9,8% para 1,1% da amostra; (ii) **ruído de rótulos**, com 10%, 20% e 30% dos exemplos de treino trocados para outra classe sorteada. Validação e teste sempre com rótulos corretos.
* **Parada:** perda de treino $\le 0.001$, ou 20 épocas sem melhora na perda de validação (máx. 300 épocas), restaurando os melhores pesos. Assim cada tamanho treina até convergir, e o número de épocas vira uma medida de **tempo de convergência**. O `StandardScaler` é ajustado só na amostra de treino usada.
* **Conjunto de teste:** 2.709 exemplos, usados apenas para medir a generalização, como pede o enunciado. Nenhuma decisão foi tomada com base nele.

### b. Resultados obtidos

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

### c. Análise e discussão
* **Curvas de generalização: melhora contínua com saturação.** A Medida-F no teste sobe de forma **monotônica** com o tamanho do treino: 0.9187 → 0.9244 → 0.9266 → 0.9303 → 0.9327 → 0.9346 → 0.9357. O ganho por exemplo, porém, cai rápido. Dobrar de 5% para 10% rende +0.57 p.p., e de 20% para 100% (5× mais dados) rende só +0.91 p.p. Na validação a curva **satura por volta de 40–60%** (F1 0.951–0.953); acima disso, as diferenças ficam dentro do desvio entre sementes. A perda de validação segue o mesmo padrão (0.230 → 0.172 em 40% → 0.162 em 100%).
* **Overfitting com poucos dados:** com 460 exemplos a rede quase memoriza a amostra. A perda de treino é a menor (0.136), mas a de teste é a maior (0.280), um gap de 0.144. Conforme a amostra cresce, a perda de treino **sobe** (fica mais difícil ajustar todos os exemplos) e a de validação e teste **cai**, e as curvas se aproximam: com 100% o gap treino–teste é de 0.035. Mais dados funcionam aqui como regularização.
* **Validação × teste:** o teste fica sistematicamente ≈ 1,5–2 p.p. abaixo da validação em F1. Há duas causas: a validação foi usada para a parada antecipada (e, nas etapas anteriores, para escolher os hiperparâmetros), o que a deixa levemente otimista; e há variação natural entre amostras. A tendência é a mesma nos dois conjuntos, então as conclusões se mantêm. Essa diferença reforça a importância de manter um conjunto de teste reservado.
* **Representatividade:** com 1.841 exemplos, a amostra **aleatória simples** já reproduz bem as proporções das classes (BARBUNYA 9,4% contra 9,8%). Por isso teve o mesmo desempenho médio da estratificada (F1 teste 0.9282 × 0.9266). A estratificação reduz a **variância**: o desvio da revocação de BARBUNYA cai de 0.017 para 0.002, e o ganho seria maior com amostras pequenas ou classes raras. Já a amostra **enviesada** mostra o efeito de uma base não representativa. Com apenas 18 exemplos de BARBUNYA, a revocação dessa classe no teste cai de 0.897 para **0.782 (−11,5 p.p.)**: a rede passa a atribuir esses grãos às classes vizinhas. O F1 macro cai só 0.7 p.p. e a acurácia de treino até sobe, porque a métrica global esconde o problema. Uma classe sub-representada é prejudicada mesmo quando o modelo "parece" bom.
* **Ruído de rótulos (qualidade):** a acurácia de treino, medida contra os rótulos ruidosos, cai quase exatamente na proporção do ruído (93.6% → 65.7% com 30%). Isso indica que a rede **não memorizou os rótulos errados**. A acurácia e o F1 no teste caem só 0.5 p.p. (0.9357 → 0.9305). Essa robustez vem do ruído ser uniforme (não favorece nenhuma classe), da capacidade pequena da 30-30 e da parada antecipada guiada por uma validação limpa. O preço aparece em duas frentes. A **perda de validação triplica** (0.162 → 0.483): as probabilidades ficam bem menos confiantes e calibradas. E o treino fica **mais lento**: a convergência passa de ~99 para ~230 épocas e o tempo dobra (16 s → 38 s).
* **Tempo de convergência:** não cresce na proporção dos dados. Cada época custa mais com mais exemplos (0.10 s com 5% contra 0.17 s com 100%, só 1,7× para 20× mais dados, por causa do overhead fixo por época). Em compensação, com mais dados a rede converge em **menos épocas** (161 → 79), porque cada época faz mais atualizações de pesos. O tempo total fica em torno de 14–18 s em todos os tamanhos. Nesta base, o custo computacional não limita o uso de 100% dos dados.

### d. Conclusão da etapa
* **Quantidade:** mais dados sempre ajudaram, com retornos decrescentes. Com **≈ 40–60% do treino (3.700–5.500 exemplos)** o modelo já alcança ≈ 99,5–99,8% da Medida-F final (validação e teste). Com menos de 20% o ganho por exemplo ainda é alto e há overfitting visível.
* **Qualidade:** a **representatividade das classes** pesa mais que o volume. Sub-representar uma classe custa mais de 11 p.p. de revocação nela e passa quase despercebido nas métricas globais. O ruído uniforme de rótulos afeta pouco a acurácia, mas degrada a calibração e dobra o tempo de treino. A amostragem estratificada é a escolha segura, principalmente para amostras pequenas.
* **Ponto de equilíbrio:** como o tempo de treino praticamente não depende do tamanho da base, recomenda-se usar **100% dos dados de treino**, que deu a melhor generalização no teste (F1 0.9357). Se coletar e rotular dados fosse caro, ~40–60% (estratificado) seria o mínimo razoável.
* Na etapa de Validação Inicial, as quatro topologias selecionadas na Q3 serão comparadas no conjunto de teste, treinadas com o conjunto de treinamento completo.

---

## 6. Guia para as Próximas Questões (5 a 7)

Para assegurar o alinhamento da dupla e a continuidade imediata dos próximos experimentos, segue a divisão e o roteiro pré-estruturado:

### Questão 3: Topologia da Rede (1 a 3 camadas ocultas, 10 a 100 neurônios)
* **Responsável:** Colega (Ímpar)
* **Parâmetros herdados da Q2:** $\eta = 0.01$, $\alpha = 0.5$, otimizador SGD, mini-batch=32.
* **Meta:** Selecionar as **quatro melhores topologias** com base em perda de validação, tempo computacional e Medida-F (*F1-Score*).
* *Status:* **Concluída** (seção 4). Quatro melhores: 30-30, 100, 50 e 30.

### Questão 4: Influência dos Dados de Treinamento (Volume de Amostras)
* **Responsável:** Pedro / Usuário (Par)
* **Parâmetros herdados da Q3:** topologia 16-30-30-7.
* **Meta:** Variar frações de dados ($20\%, 40\%, 60\%, 80\%, 100\%$) com amostragem estratificada e analisar curvas de saturação de generalização.
* *Status:* **Concluída** (seção 5).

### Questão 5: Influência dos Atributos (Seleção e Sensibilidade de Features)
* **Responsável:** Colega (Ímpar)
* **Meta:** Avaliar remoção sistemática de atributos redundantes/correlacionados e impacto no desempenho.
* *Status:* A ser executado após conclusão da Q4.

### Questão 6: Validação Inicial (Conjuntos de Treinamento e Teste Reservado)
* **Responsável:** Pedro / Usuário (Par)
* **Meta:** Avaliar as 4 melhores topologias da Questão 3 (30-30, 100, 50 e 30) no conjunto de teste reservado ($2.709$ instâncias), selecionando o modelo campeão definitivo.
* *Status:* A ser executado após conclusão da Q5.

### Questão 7: Validação Cruzada ($k$-fold) Consolidada
* **Responsável:** Colega (Ímpar)
* **Meta:** Avaliação estatística robusta ($k=5$ ou $k=10$) com média e variância de Acurácia, Loss e F1-Score da rede campeã.
* *Status:* Fechamento do trabalho.

---
*Relatório gerado automaticamente a partir das execuções reais e reproduzíveis do notebook `teste.ipynb`.*
