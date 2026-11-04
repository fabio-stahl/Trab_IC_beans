### Relato Técnico — Questão 3

#### a. Descrição da configuração experimental
* **Faixas testadas:** 1, 2 e 3 camadas ocultas × {10, 20, …, 100} neurônios por camada (grade uniforme, 30 redes), mais 5 topologias em funil (64-32, 100-50, 60-30-15, 80-40-20, 100-50-25). São **35 topologias**, de 247 a 22.607 parâmetros.
* **Ativações e perda:** ReLU nas camadas ocultas (inicialização He Normal), Softmax na saída (Glorot Uniform), Entropia Cruzada Categórica.
* **Critério de parada:** perda de treino $\le 0.001$ **ou** 100 épocas.
* **Fixos:** SGD com $\eta = 0.01$ e $\alpha = 0.5$ (melhores da Q2), lote 32, semente 42, mesmos conjuntos de treino (9.208) e validação (1.626). O teste não foi usado.
* **Seleção:** as 6 topologias com menor perda de validação foram re-treinadas com mais 3 sementes (4 no total); o ranking usa a perda média de validação.

#### b. Resultados obtidos
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

#### c. Análise e discussão
* **Critério de parada:** nenhuma topologia chegou a perda $\le 0.001$. Como na Q2, as classes se sobrepõem (sobretudo SIRA × DERMASON e BARBUNYA × CALI), e a perda de treino fica entre 0.14 e 0.20. Todas as redes rodaram as 100 épocas.
* **Complexidade × desempenho:** mais capacidade **reduz a perda de treino** (0.1965 com 1×10 contra 0.1398 com 3×100), mas **não melhora a validação**. A perda de validação de todas as 35 redes fica numa faixa estreita (0.157–0.179) e a Medida-F entre 0.945 e 0.957. As melhores redes são as **pequenas e médias** (1 camada com 30–100 neurônios, ou 2×30). Acrescentar camadas piora a validação: a média da perda de validação é ≈ 0.162 com 1 camada, ≈ 0.165 com 2 e ≈ 0.173 com 3.
* **Underfitting:** nenhuma rede ficou acima do limiar (perda > 0.20 no treino **e** na validação). As de menor capacidade (1×10, 2×10 e 1×20, com perda de treino de 0.19–0.20) estão no limite: aprendem mais devagar e ainda descem na época 100. Mesmo assim, generalizam tão bem quanto as maiores (F1 val ≈ 0.951–0.954). Para 16 atributos e 7 classes, 10 neurônios já bastam.
* **Overfitting:** aparece nas redes **profundas e largas**. Com 3 camadas e ≥ 20 neurônios, a perda de validação chega ao mínimo cedo (épocas 26–47) e depois **sobe**, enquanto a de treino continua caindo. O caso mais claro é o **100-100-100**: treino 0.1398, validação 0.1793, mínimo de 0.1685 na época 32. As redes de 1 camada não mostram esse efeito: a validação ainda cai na época 100.
* **Tempo de treinamento:** de 247 para 22.607 parâmetros (≈ 90×), o tempo passou só de ~16 s para ~21–24 s. Nesta base pequena, em CPU e com lote 32, o custo vem quase todo do overhead fixo de cada passo (≈ 288 lotes por época), e não das multiplicações de matriz. Cada camada extra acrescenta ≈ 1 s em 100 épocas.
* **Equilíbrio:** redes maiores custam um pouco mais de tempo e não melhoram a generalização. Pelo contrário, ficam mais propensas a overfitting. O ponto de equilíbrio está em redes com **≈ 700 a 2.500 parâmetros** (1 camada com 30–100 neurônios, ou 2×30).
* **Estabilidade:** entre as 6 finalistas, a perda média de validação varia só 0.0013, menos que o desvio entre sementes (até 0.0027). Na prática, estão **estatisticamente empatadas**. O desempate considera também a variabilidade e a Medida-F.

#### d. Seleção das quatro melhores topologias
1. **30-30 (2 camadas):** menor perda média de validação (0.1609) e a menor perda numa única execução de toda a grade (0.1572), com F1 de 0.9545. Rede pequena (1.657 parâmetros), sem sinal de overfitting em 100 épocas. **Foi a escolhida para a Questão 4.**
2. **100 (1 camada):** praticamente a mesma perda (0.1610) com desvio entre sementes muito baixo (± 0.0009). É uma rede larga e rasa, ainda em melhora na época 100.
3. **50 (1 camada):** a **mais estável** (perda val ± 0.0004) e com a **maior Medida-F média** (0.9549), usando só 1.207 parâmetros.
4. **30 (1 camada):** a mais econômica (727 parâmetros), com desempenho quase igual ao das maiores (F1 0.9536 ± 0.0005). Mostra que redes muito pequenas já bastam para o problema.

As topologias de 3 camadas e em funil ficaram fora: todas tiveram perda de validação maior e indícios de overfitting.

#### e. Conclusão da etapa
* Na Dry Bean Dataset, a **arquitetura tem efeito pequeno na generalização**: as 35 topologias ficam numa faixa de apenas 1,2 p.p. de Medida-F (0.945–0.957). A topologia pesa menos que a escolha de $\eta$ e $\alpha$ (Q2).
* **Mais camadas e mais neurônios** reduzem a perda de treino, mas antecipam o mínimo da validação e geram overfitting, sem ganho de Medida-F. **Uma ou duas camadas pequenas** dão o melhor compromisso entre generalização, estabilidade e custo.
* As quatro configurações selecionadas (**30-30, 100, 50 e 30**) serão comparadas no conjunto de teste na etapa de Validação Inicial. A **30-30** é usada na Questão 4.
