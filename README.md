# PathoShift-CRC Mini Challenge — Processamento robusto de imagens histopatológicas

Trabalho final de Processamento de Imagens (UFRRJ). Pipeline de classificação de tiles H&E
(9 classes do NCT-CRC-HE-100K-NONORM) com **baseline reaproveitado da Entrega 1** e uma
**proposta de pré-processamento** avaliada sob o protocolo oficial (`train.csv`,
`validation.csv`, `robustness.csv`).

> Uso exclusivamente acadêmico. Não é ferramenta de diagnóstico e não deve apoiar decisão clínica.

## Pergunta de investigação

> *Padronizar a imagem antes da extração de atributos (normalização de cor e/ou suavização)
> torna o classificador mais robusto a mudanças de aquisição (brilho/contraste, JPEG, desfoque,
> ruído) do que treinar com aumentação dessas mudanças? E as duas estratégias se somam?*

Métrica principal: **F1-macro** (classes balanceadas, mas o enunciado exige métricas por classe).
Também são reportados acurácia, F1/precisão/revocação por classe, matriz de confusão, ECE
(calibração) e a **queda de F1-macro entre validação e robustez**.

## Resultados (avaliador oficial)

| Modelo | F1-macro validação | F1-macro robustez | Queda | ECE val / rob |
|---|---|---|---|---|
| Baseline (`raw`, C=10) | 0.963 | 0.602 | 0.361 | 0.017 / 0.196 |
| **Proposta (`gauss1+aug`, C=10)** | **0.966** | **0.925** | **0.041** | 0.012 / 0.014 |

F1-macro por perturbação (baseline → proposta): brilho/contraste 0.33 → 0.83 · desfoque 0.69 → 0.96 ·
ruído 0.38 → 0.94 · JPEG 0.89 → 0.97. A aumentação de aquisição é o fator principal; a suavização
gaussiana soma ganho; as normalizações de cor (Reinhard/Macenko) não ajudaram (ver `report/report.pdf`).

## Pipeline

```
TIFF 224×224 ──► [perturbação oficial, só p/ robustness.csv]
              ──► pré-processamento (variante)       ◄── objeto de estudo
                    ├─ suavização: nenhuma | Gaussiana σ=1 | mediana 3×3
                    └─ cor: nenhuma | Reinhard (LAB) | Macenko (densidade óptica)
              ──► atributos (152-D): momentos+histogramas RGB/HSV, GLCM, LBP uniforme (P=24,R=3)
              ──► StandardScaler + SVM RBF (C ∈ {1,10}, gamma='scale', Platt p/ confiança)
```

| Configuração | Pré-processamento | Treino |
|---|---|---|
| **baseline** (`raw`) | nenhum | `train.csv` |
| `reinhard`, `macenko` | normalização de cor | `train.csv` |
| `gauss1`, `median3` | redução de ruído | `train.csv` |
| `reinhard+gauss1`, `macenko+gauss1` | suavização → normalização | `train.csv` |
| `<qualquer>+aug` | idem | `train.csv` + 1 cópia aumentada por imagem |

A aumentação (`random_acquisition_shift`) sorteia uma mudança genérica por imagem, com
faixas fixas definidas a priori: brilho U(−0.2, 0.2), contraste U(0.75, 1.3), matiz ±6,
saturação ×U(0.8, 1.2); JPEG q∈[20, 90]; desfoque σ∈U(0.4, 2.0); ruído σ∈U(3, 20).

### Protocolo (sem vazamento)

1. **Treino**: somente `train.csv` (9.000 imagens).
2. **Seleção** de configuração e de `C`: somente em `validation.csv`, em duas versões:
   limpa e **validação-proxy** (as mesmas 2.250 imagens com mudanças aleatórias geradas pelo
   nosso gerador, seed `2000·10⁵+i`, independente de `robustness.csv`).
   Regra fixada antes de olhar a robustez: *baseline* = `raw` com C de maior F1 na validação
   limpa; *proposta* = configuração com maior média (F1 validação, F1 proxy).
3. **Avaliação final**: a seleção é gravada em `outputs/results/selection.json`; só então
   `robustness.csv` é avaliado. Nada é retreinado ou escolhido com base nele. As linhas de
   ablação em `robustness.csv` são reportadas apenas como análise.
4. Rótulos vêm da coluna `label` dos manifestos; nomes de arquivo não são usados como atributo.

### Perturbações de `robustness.csv`

Implementadas em `src/pathoshift/preprocessing/perturbations.py`, determinísticas pela `seed` de cada linha:

| tipo | parâmetros | implementação |
|---|---|---|
| `brightness_contrast` | brightness=0.15; contrast=1.20 | `y = clip(contrast·x + 255·brightness)` (convenção α/β do OpenCV/Albumentations — fórmula A) |
| `jpeg` | quality=35 | codificação/decodificação JPEG com Pillow |
| `gaussian_blur` | sigma=1.2 | `cv2.GaussianBlur`, kernel automático, borda reflect-101 |
| `gaussian_noise` | stddev=12 | `x + N(0, 12)` com `numpy.random.default_rng(seed)`, arredondado e recortado em [0, 255] |

**Sensibilidade à fórmula de `brightness_contrast` (sem retreino):** como o manifesto não especifica a equação exata de `brightness_contrast`, o subcomando `python -m pathoshift.challenge sensitivity` avalia os modelos congelados (treinados com a fórmula A) sob quatro convenções possíveis: **A** (`c·x + 255·b`, padrão nos CSVs entregues), **B** (`(x − 128)·c + 128 + 255·b`), **C** (`(x − μ)·c + μ + 255·b`) e **D** (brilho multiplicativo `1.15·x` seguido de contraste `×1.2` estilo PIL/torchvision). A fórmula A é o caso mais severo de saturação em 255 (95,8% dos pixels em ADI); nas leituras B, C e D o F1-macro da proposta no subconjunto `brightness_contrast` sobe de **0.830** (A) para **0.896** (B), **0.890** (C) e **0.843** (D) (F1-macro de robustez global de **0.925** em A e **0.929–0.941** em B–D, contra **0.602–0.649** do baseline).

## Instalação

Requisitos: Python ≥ 3.10 (testado em 3.13.7), Node.js ≥ 18 (para o avaliador oficial).
Não precisa de GPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # ou defina PYTHONPATH=src
```

Os dados **não** acompanham a entrega. Descompacte o pacote de imagens da disciplina
(`PathoShift-CRC-images.zip`) em uma pasta, por exemplo `../dados`, de modo que existam
`../dados/manifests/*.csv` e `../dados/NCT-CRC-HE-100K-NONORM/<CLASSE>/*.tif`.

## Reprodução completa

```bash
# 1) atributos de todas as variantes (cache em outputs/features; ~80 min em 4 núcleos/8 threads)
python -m pathoshift.challenge features   --data-root ../dados --output outputs --jobs 8
# 2) análise inicial dos dados (figuras em outputs/eda)
python -m pathoshift.challenge eda        --data-root ../dados --output outputs --jobs 8
# 3) grade/seleção na validação, modelos finais, robustez, figuras e CSVs oficiais
python -m pathoshift.challenge experiment --data-root ../dados --output outputs --deliverable-dir .
# 4) análise de sensibilidade às convenções A/B/C/D de brightness_contrast (sem retreino)
python -m pathoshift.challenge sensitivity --data-root ../dados --output outputs --jobs 8
# 5) avaliador oficial
node ../kit/evaluate_pathoshift_crc.mjs --manifest ../dados/manifests/validation.csv --predictions predictions_validation.csv --output outputs/results/eval_validation.json
node ../kit/evaluate_pathoshift_crc.mjs --manifest ../dados/manifests/robustness.csv  --predictions predictions_robustness.csv  --output outputs/results/eval_robustness.json
```

### Somente inferência (modelo já treinado)

```bash
python -m pathoshift.challenge predict --data-root ../dados --model outputs/models/proposal.joblib --split validation --out-csv predictions_validation.csv
python -m pathoshift.challenge predict --data-root ../dados --model outputs/models/proposal.joblib --split robustness --out-csv predictions_robustness.csv
```

A inferência extrai atributos de uma única variante e leva poucos minutos por manifesto
(limite do enunciado: 10 min com 4 núcleos e 8 GiB de RAM).

## Configuração principal e seeds

| item | valor |
|---|---|
| seed global | `42` (numpy, SVM/Platt, amostragem da referência Reinhard) |
| seeds aumentação / proxy | `1000·10⁵ + i` / `2000·10⁵ + i` (i = linha do manifesto) |
| seeds robustez | coluna `seed` de `robustness.csv` |
| referência Reinhard | média das estatísticas LAB de 900 imagens de treino (100/classe), em `outputs/features/reinhard_target.json` |
| referência Macenko | vetores H&E e concentrações padrão de Macenko et al. (2009) |
| classificador | `StandardScaler` + `SVC(kernel='rbf', gamma='scale', C∈{1,10}, probability=True)` |

## Saídas

| caminho | conteúdo |
|---|---|
| `predictions_validation.csv`, `predictions_robustness.csv` | CSVs oficiais (proposta selecionada) |
| `outputs/predictions/baseline/` | mesmos CSVs para o baseline |
| `outputs/results/summary.md`, `summary.json` | métricas, queda de F1, por perturbação, por classe, ECE, hardware e tempos |
| `outputs/results/grid_validation.csv`, `ablation.csv`, `selection.json` | grade, ablação e seleção congelada |
| `outputs/eda/` | distribuição de classes, exemplos, cor/foco por classe, exemplos de perturbação |
| `outputs/figures/` | matrizes de confusão, F1 por condição/classe, ablação, calibração, galerias de erros/acertos |
| `outputs/models/*.joblib` | modelos baseline e proposta |
| `report/report.pdf` | relatório |

## Estrutura e aproveitamento da Entrega 1

| componente | origem |
|---|---|
| `preprocessing/stain_norm.py` (Macenko, Reinhard), `preprocessing/quality.py` (tecido, foco) | Entrega 1, sem alteração |
| `models/classical.py` (atributos cor+GLCM+LBP) | Entrega 1, sem alteração; classificador passou a ser SVM via `challenge.py` |
| `training/metrics.py`, `visualization/*` | Entrega 1 (mantidos) |
| `data/official.py` | **novo**: leitura e checagem dos manifestos oficiais |
| `preprocessing/perturbations.py` | **novo**: perturbações de `robustness.csv` e aumentação |
| `preprocessing/pipeline.py` | **novo**: pré-processamento configurável (ablação) |
| `challenge.py`, `figures.py` | **novos**: protocolo oficial, seleção, robustez, figuras, CSVs |
| `*/__init__.py` | **modificados**: PyTorch passou a ser opcional |
| `cli.py`, `data/manifest.py`, `data/splits.py`, `models/cnn.py`, `training/trainer.py` | Entrega 1; **não usados** na entrega final (partições próprias substituídas pelos manifestos oficiais) |

Testes: `pytest tests/` (inclui `tests/test_official.py`, que cobre o determinismo das perturbações e o contrato do CSV).
