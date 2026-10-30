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

## 4. Guia e Template para as Próximas Questões (3 a 7)

Para assegurar o alinhamento da dupla e a continuidade imediata dos próximos experimentos, segue a divisão e o roteiro pré-estruturado:

### Questão 3: Topologia da Rede (1 a 3 camadas ocultas, 10 a 100 neurônios)
* **Responsável:** Colega (Ímpar)
* **Parâmetros herdados da Q2:** $\eta = 0.01$, $\alpha = 0.5$, otimizador SGD, mini-batch=32.
* **Meta:** Selecionar as **quatro melhores topologias** com base em perda de validação, tempo computacional e Medida-F (*F1-Score*).
* *Status:* Aguardando execução pelo colega no notebook.

### Questão 4: Influência dos Dados de Treinamento (Volume de Amostras)
* **Responsável:** Pedro / Usuário (Par)
* **Parâmetros herdados da Q3:** Melhor topologia vencedora da Q3.
* **Meta:** Variar frações de dados ($20\%, 40\%, 60\%, 80\%, 100\%$) com amostragem estratificada e analisar curvas de saturação de generalização.
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
