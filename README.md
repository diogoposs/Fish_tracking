# Fish Tracking

Projeto acadêmico de rastreamento de zebrafish (_Danio rerio_) e normalização de iluminação em vídeos, desenvolvido no contexto do trabalho de conclusão de curso em Engenharia de Controle e Automação da UFSM, com aplicação em pesquisas do LANE.

**Tecnologias:** Python, OpenCV, PyTorch, torchvision, NumPy, Matplotlib e scikit-image.

## Objetivo

Investigar o uso de uma U-Net como pré-processamento para reduzir variações de iluminação e apoiar o rastreamento por segmentação de cor HSV.

## Etapas do processamento

1. Treinar a U-Net com pares de imagens de entrada e referência.
2. Aplicar o modelo aos frames de um vídeo e gerar um vídeo normalizado.
3. Selecionar áreas válidas e sete pontos do peixe no primeiro frame.
4. Segmentar por HSV, aplicar filtro de mediana, selecionar o maior contorno e calcular seu centroide.
5. Verificar deslocamentos e exportar coordenadas, estados de detecção e gráficos.

O rastreador também pode receber um vídeo diretamente, sem executar as etapas de treinamento e normalização.

## Arquivos

| Arquivo                                                | Responsabilidade                                            |
| ------------------------------------------------------ | ----------------------------------------------------------- |
| [ilumination_unet_train.py](ilumination_unet_train.py) | Dataset pareado, arquitetura U-Net, treinamento e avaliação |
| [video_normalizer.py](video_normalizer.py)             | Inferência por frame e gravação de vídeo normalizado        |
| [Tracking.py](Tracking.py)                             | Seleção de ROIs e pontos, rastreamento HSV, CSV e gráficos  |
| [Relatório do TCC](Relatório_TCC_Diogo_Possebon.pdf)   | Documentação acadêmica do projeto                           |

## Preparar o ambiente

É necessário Python com interface gráfica para as janelas do OpenCV. O treinamento usa CUDA quando disponível e CPU nos demais casos.

```bash
git clone https://github.com/diogoposs/Fish_tracking.git
cd Fish_tracking
python -m venv .venv
```

Ative o ambiente:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Para treinamento com GPU, utilize uma instalação de PyTorch compatível com o seu ambiente CUDA.

## Executar

Os scripts atuais utilizam caminhos configurados no próprio código. Antes de executar, ajuste:

| Script                      | Configurações                                          |
| --------------------------- | ------------------------------------------------------ |
| `ilumination_unet_train.py` | `input_folder` e `target_folder`, dentro de `main()`   |
| `video_normalizer.py`       | `model_path`, `video_input` e `video_output`           |
| `Tracking.py`               | `caminho_video`, `DIST_MAX` e `LIMITE_REINICIALIZACAO` |

### Treinamento

Os arquivos de entrada devem começar com `frame` e conter `_br`. O trecho anterior a `_br` identifica a imagem de referência, que pode ter extensão JPG ou PNG.

Exemplo: `frame001_br120.png` é associado a `frame001.jpg` ou `frame001.png` na pasta de referência.

```bash
python ilumination_unet_train.py
```

A versão publicada usa imagens de 192 × 192, batch 8, até 15 épocas, Adam com taxa de aprendizado de 0,001, perda L1 e early stopping após três épocas sem melhora de validação. A divisão é aproximadamente 70% treino, 20% validação e 10% teste.

### Normalização do vídeo

```bash
python video_normalizer.py
```

Use um checkpoint compatível com a arquitetura publicada. A inferência ocorre em 192 × 192; a saída é redimensionada à resolução original e gravada com o FPS informado pelo vídeo de entrada.

### Rastreamento

```bash
python Tracking.py
```

Desenhe uma ou mais ROIs com o mouse, pressione **n** e selecione sete pontos sobre o peixe. Pressione **q** para encerrar. Os resultados são gravados em `Dados_rastreamento/`.

## Saídas

- Checkpoints, exemplos de entrada/saída/referência, curvas de perda e métricas PSNR/SSIM do treinamento.
- Vídeo normalizado.
- CSV com as colunas `Frame,X,Y,Distancia,Status`, gravado a cada frame processado.
- Gráficos de deslocamento e trajetória.

O estado `falso_positivo` é atribuído pela regra de deslocamento do rastreador; não corresponde a uma comparação automática com anotações de referência.

## Limitações de reprodução

Os vídeos, o dataset pareado e os pesos treinados não acompanham o repositório. O processamento depende da iluminação, das ROIs e dos pontos selecionados; não há garantia de desempenho fora das condições avaliadas.

A divisão do dataset ocorre por pares de imagens. Para avaliar generalização entre vídeos, uma evolução metodológica é separar as gravações de origem antes de gerar suas variações, evitando sobreposição entre treino e avaliação.

O projeto não inclui uma suíte automatizada de testes.
