# Modelo de Propensão — Bank Marketing

Modelo que indica **para quais clientes um banco deve ligar primeiro** para vender um depósito a prazo. Com a mesma verba de ligações, o modelo gera **cerca de 49% mais lucro** do que ligar ao acaso.

Projeto de estudo construído de ponta a ponta: EDA, preparação dos dados, seleção de variáveis, modelagem, tunagem, validação e visões de negócio.

---

## Sumário

1. [Problema de negócio](#1-problema-de-negócio)
2. [Base de dados](#2-base-de-dados)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Desenvolvimento](#4-desenvolvimento)
5. [Resultados](#5-resultados)
6. [Visões de negócio](#6-visões-de-negócio)
7. [Principais insights](#7-principais-insights)
8. [Conclusões e limitações](#8-conclusões-e-limitações)
9. [Próximos passos](#9-próximos-passos)
10. [Como executar](#10-como-executar)
11. [Tecnologias](#11-tecnologias)

---

## 1. Problema de negócio

Um banco português fez campanhas de telemarketing para oferecer um **depósito a prazo**. Ligar para todos os clientes custa caro, e a maioria não contrata (a conversão média é de 11,3%).

**Objetivo:** estimar a probabilidade de cada cliente contratar o produto depois de uma ligação, para que o banco:

- priorize as ligações para quem tem mais chance de converter;
- gaste menos ligações por venda;
- encontre o ponto em que ligar para mais clientes deixa de dar lucro.

---

## 2. Base de dados

- **Fonte:** [Bank Marketing — Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- **Origem:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Moro, Cortez e Rita, 2014)
- **Arquivo usado:** `bank-additional-full.csv` (separador `;`)
- **Tamanho:** 41.188 ligações × 21 colunas, de maio de 2008 a novembro de 2010
- **Target:** `y`, se o cliente contratou o depósito (`yes` = 1, `no` = 0)

| Grupo | Variáveis |
|---|---|
| Perfil do cliente | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Último contato da campanha | `contact`, `month`, `day_of_week`, `duration` |
| Histórico de contatos | `campaign`, `pdays`, `previous`, `poutcome` |
| Contexto econômico | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

> **Atenção:** `duration` (duração da ligação) só é conhecida depois do contato. Ela foi removida para evitar vazamento de informação.

---

## 3. Estrutura do repositório

```
├── notebooks/
│   ├── 01_EDA_bank_marketing.ipynb              # estudo de público e metadados
│   ├── 02_data_prep_bank_marketing.ipynb        # limpeza, variáveis novas, split, encoding
│   ├── 03_feature_selection_bank_marketing.ipynb # seleção de variáveis
│   ├── 04_modelagem_bank_marketing.ipynb        # modelos, tunagem e validação
│   └── 05_escoragem_bank_marketing.ipynb        # escoragem de uma base nova
├── excel/
│   └── tabelas_simulacao_financeira.xlsx        # faixas, decis e simulação financeira
├── imagens/                                     # gráficos usados neste README
└── README.md
```

---

## 4. Desenvolvimento

### 4.1 EDA — estudo de público

- **Metadados:** tipos, preenchimento, nulos e `unknown` (o "nulo disfarçado" da base), cardinalidade e duplicatas (12 linhas).
- **Tabela padrão por variável:** volume, % do público, taxa de conversão, lift, WoE e IV, para categóricas e faixas das numéricas.
- **Correlações:** Pearson, Spearman, ponto-bisserial com o target e Cramér's V.
- **Visão temporal:** o ano foi reconstruído pela ordem das linhas (`AnoMes`), revelando que **o público muda muito no tempo**.

![Volume e conversão por período](imagens/eda_periodo.png)

| Período | Clientes | Conversão |
|---|---|---|
| 2008-05/06 | ~12.100 | 3,5% |
| 2008-07/08 | ~11.800 | 5,7% |
| 2008-10/12 | ~3.700 | 6,3% |
| 2009-03/05 | ~8.500 | 12,8% |
| 2009-06/08 | ~1.700 | 35,7% |
| 2009-09/12 | ~1.200 | 43,6% |
| 2010-03/06 | ~900 | 54,2% |
| 2010-07/11 | ~1.200 | 50,6% |

![Correlação de Pearson](imagens/eda_correlacao.png)

Os indicadores econômicos são quase a mesma informação (Euribor × variação do emprego: 0,97; nº de empregados × Euribor: 0,95).

### 4.2 Preparação dos dados

- Remoção de duplicatas e da variável `duration` (vazamento).
- **Variáveis novas:** `flag_nunca_contatado` e `pdays_tratado` (o valor 999 de `pdays` significa "nunca contatado"), `flag_sucesso_anterior`, `flag_celular`, `contatos_totais`, `qtd_produtos`, `faixa_idade`, `log_campaign`.
- **Split OOS:** treino e teste 70/30, **estratificado por target e ano** (28.823 e 12.353 linhas).
- **Tudo ajustado só no treino e aplicado no teste:** imputação (mediana nas numéricas, moda nos `unknown`), Target Encoding com cross-fitting e padronização (StandardScaler).

> Modelos de boosting não precisam de padronização. A etapa foi mantida para treinar o pipeline completo de preparação.

**Por que só OOS, sem OOT?** Na primeira versão, o OOT (treino até mai/2009, teste depois) treinava o modelo apenas com o público de baixa conversão. Como a taxa de conversão e o perfil do público mudam muito entre os períodos, a validação ficou só fora da amostra (OOS): treino e teste incluem todos os períodos, e o modelo aprende a separar alta e baixa propensão tanto no público bom quanto no ruim.

### 4.3 Seleção de variáveis

| Etapa | Regra |
|---|---|
| Regra de negócio | Saem as variáveis de calendário (`month`, `trimestre`) e de contexto econômico, que identificam o período da ligação e não o cliente |
| Missing | Mais de 70% de nulos/`unknown` antes da imputação |
| Variância | 99% ou mais das linhas com o mesmo valor |
| Colinearidade | Spearman acima de 0,8 (fica a de maior IV) e VIF acima de 10 (sai a de menor IV) |
| Feature importance | Permutation importance com LightGBM em 5 folds; sai a variável com importância média ≤ 0 ou que ajudou em menos de 3 dos 5 folds |

**Variáveis finais:** `age`, `log_campaign`, `qtd_produtos`, `te_job`, `te_marital`, `te_education`, `te_default`, `te_loan`, `te_contact`.

> A permutation importance mede quanto a AUC cai, em dados que o modelo não viu, quando uma variável é embaralhada. Uma variável de ruído aleatório incluída no teste teve 15% do *gain* do modelo, mas permutation importance perto de zero, e por isso a decisão usa a permutation.

### 4.4 Modelagem e tunagem

- **Modelos comparados:** Random Forest, Gradient Boosting, LightGBM, XGBoost e CatBoost.
- **Métricas:** KS, AUC e Gini (ordenação) e Recall, Precision, F1 e Acurácia (com o corte do KS do treino).
- **Regra de overfitting:** queda de Gini do treino para o teste até 5 p.p. = OK, de 5 a 10 = atenção, acima de 10 = overfitting.
- **Tunagem do LightGBM com Optuna:**
  - validação cruzada de 5 folds só no treino;
  - early stopping por logloss;
  - restrição monotônica (`te_*` só aumentam o score, `log_campaign` só diminui);
  - nota = AUC de validação − penalidade pelo gap treino-validação acima de 0,03 − penalidade por inversões nos decis out-of-fold;
  - parada antecipada quando a AUC não melhora por 50 trials.

---

## 5. Resultados

### Overfitting: treino × teste

| Modelo | AUC treino | AUC teste | Gini treino | Gini teste | Queda de Gini | Status |
|---|---|---|---|---|---|---|
| LightGBM padrão | 0,868 | 0,727 | 0,736 | 0,454 | 28,2 p.p. | Overfitting |
| Gradient Boosting | 0,757 | 0,734 | 0,515 | 0,468 | 4,6 p.p. | OK |
| **LightGBM tunado** | **0,748** | **0,737** | **0,496** | **0,474** | **2,2 p.p.** | **OK** |

**KS do modelo final:** 0,358 no treino e 0,345 no teste.

### Ordenação por decil (teste)

![Taxa de conversão por decil](imagens/decis.png)

| Decil | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Conversão | 39,5% | 14,6% | 12,7% | 11,7% | 9,9% | 6,6% | 5,5% | 4,9% | 3,7% | 3,6% |

O decil de maior score converte **3,5 vezes a média** (11,3%), sem inversões.

### Estabilidade no tempo

![KS treino × teste no tempo](imagens/ks_tempo.png)

Treino e teste andam juntos em todos os meses: não há overfitting por safra. A separação é menor em 2008 e maior a partir de 2009.

---

## 6. Visões de negócio

### 6.1 Faixas de propensão

Cortes definidos no treino, na escala `score_1000` (score × 1000):

| Faixa | Regra | % do treino | Conversão no treino |
|---|---|---|---|
| Alta | acima de 337 | 5,0% | 57,4% |
| Média | de 58 a 337 | 62,6% | 11,3% |
| Baixa | até 57 | 32,4% | 4,1% |

![Taxa de conversão por faixa no tempo](imagens/faixas_taxa.png)
![Distribuição das faixas no tempo](imagens/faixas_distribuicao.png)

Na base de teste, de 2009 a 2010, a ordem **Alta > Média > Baixa** se mantém em todos os períodos. A faixa Alta converte de 40% a 71%.

### 6.2 Simulação financeira e corte ótimo

**Premissas:** R$ 10 por ligação, R$ 200 de receita por depósito fechado e taxas de conversão da base de teste.

![Corte ótimo — lucro acumulado com × sem modelo](imagens/corte_otimo.png)

- Vale ligar para um decil enquanto **taxa de conversão > custo ÷ receita** (aqui, 5%).
- O lucro máximo vem ao ligar até o **decil 7 (70% da base)**: **R$ 161.730**.

| Comparação | Sem modelo | Com modelo (70% da base) | Ganho |
|---|---|---|---|
| Mesmas 8.647 ligações, ao acaso | R$ 108.433 | R$ 161.730 | **+R$ 53.297 (+49,2%)** |
| Ligar para a base inteira (12.353 ligações) | R$ 154.907 | R$ 161.730 | **+R$ 6.823 (+4,4%) com 30% menos custo (−R$ 37.060)** |

---

## 7. Principais insights

1. **O público muda muito no tempo.** A conversão sai de 3,5% em 2008 para mais de 50% em 2010, com cada vez menos ligações.
2. **AUC alta nem sempre significa um bom modelo.** As variáveis de calendário e de contexto econômico aumentavam a AUC, mas o modelo aprendia *quando* a ligação aconteceu, e não *quem* era o cliente. Na hora de usar o score, todos os clientes de uma mesma campanha estão no mesmo mês, então essas variáveis não ajudam a decidir para quem ligar.
3. **O histórico de contato e o canal são os maiores sinais do cliente.** Quem já contratou antes e quem é contatado pelo celular converte mais.
4. **Volume importa na leitura das visões.** Células com poucos clientes (por exemplo, a faixa Baixa em 2010) oscilam por acaso. Agrupar meses vizinhos deixa as visões confiáveis.
5. **O modelo vale mais quando a verba é limitada.** Ligar para todos dá o mesmo resultado com ou sem modelo. O ganho está em priorizar.

---

## 8. Conclusões e limitações

**Conclusões**

- O modelo **ordena bem os clientes**: decis sem inversões, faixas ordenadas em todos os períodos e treino e teste estáveis.
- Não há overfitting: a queda de Gini do treino para o teste é de 2,2 p.p.
- Com a mesma verba, o modelo gera **cerca de 49% mais lucro** do que ligar ao acaso, e atinge o lucro máximo com 30% menos ligações do que contatar todos.

**Limitações**

- **O poder de separação é moderado dentro de um mesmo período**, principalmente em 2008. Parte da separação geral vem da diferença entre o público de 2008 e o de 2009–2010.
- **A validação é só OOS.** Treino e teste vêm dos mesmos períodos. Um público futuro muito diferente pode reduzir o desempenho.
- **A simulação financeira depende de premissas** de custo e receita e das taxas observadas no teste. É uma estimativa.
- **A distribuição das faixas muda no tempo** (a Alta vai de 4% para mais de 40% do público). Com cortes fixos, a mesma faixa representa fatias diferentes da base.

---

## 9. Próximos passos

- Monitorar o **PSI do score** e o **KS por período** em produção.
- Recalibrar os cortes das faixas quando o público mudar.
- Buscar novas variáveis do cliente (relacionamento com o banco, produtos, canal preferido) para melhorar a ordenação dentro de cada período.
- Testar uma política de insistência por faixa (número máximo de tentativas por cliente).

---

## 10. Como executar

1. Baixe o `bank-additional-full.csv` no [Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) e salve na pasta `Base/`.
2. Ajuste os caminhos na célula de setup de cada notebook (o projeto foi feito no Google Colab, com os dados no Google Drive).
3. Rode os notebooks na ordem: `01` → `02` → `03` → `04`.
4. Para escorar uma base nova no formato original, rode o `05_escoragem`. Ele usa os artefatos salvos (`transformadores_prep.joblib`, `variaveis_selecionadas.json` e `lgbm_tunado.joblib`) e devolve o score e a faixa de propensão de cada cliente, sem treinar nada.

---

## 11. Tecnologias

- **Python:** pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, statsmodels, SciPy
- **Visualização:** Matplotlib, Seaborn e Excel
- **Ambiente:** Google Colab e Google Drive

---

**Autor:** Murilo · [LinkedIn](#) · [GitHub](#)
