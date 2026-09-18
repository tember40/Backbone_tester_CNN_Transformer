# CIFAR-10 Multiclass Classification Framework

이 프로젝트는 CIFAR-10 데이터셋을 활용하여 퍼셉트론과 MLP의 기초부터 AlexNet, 현대 CNN, Vision Transformer까지 직접 실행·비교하는 교육용 이미지 분류 프레임워크입니다.

## 웹 학습 자료

[웹 학습 자료 바로가기](https://tember40.github.io/Backbone_tester_CNN_Transformer/)에서 설치 없이 개념 설명, 실제 프로젝트 코드, 단계별 계산, 결정경계 시각화와 확인 문제를 사용할 수 있습니다.

현재 1장 **퍼셉트론**과 2장 **선형 분리와 XOR**을 공개했습니다. 논문과 역사적 맥락, 수식, 실제 프로젝트 코드, 단계별 시각화, 확인 문제를 같은 흐름으로 구성했습니다. 웹 실험은 브라우저에서 가볍게 실행하고, 실제 PyTorch 실습은 노트북으로 이어집니다.

저장소를 내려받은 뒤 웹 학습 자료를 로컬에서 확인하려면 별도 패키지 설치 없이 다음 명령만 실행합니다.

```powershell
# Windows PowerShell
py -3 -m http.server 8000 --directory docs
```

```bash
# macOS / Linux
python3 -m http.server 8000 --directory docs
```

브라우저에서 `http://localhost:8000`을 열면 됩니다. `main` 브랜치의 `docs/` 변경 사항은 GitHub Pages 배포 작업을 통해 자동으로 반영됩니다.

## 프로젝트 구조

```
classification
 ├── Cifar-10 dataset/  # 기존 개발 환경의 데이터셋(있으면 자동 재사용)
 ├── backbone/          # 다양한 백본 모델들의 파이썬 코드
 │   ├── VGG.py
 │   ├── ResNet.py
 │   ├── MobileNet.py
 │   ├── EfficientNet.py
 │   ├── InceptionV3.py
 │   ├── ConvNeXt.py
 │   ├── SeNet.py
 │   ├── ViT.py
 │   └── PVT.py
 ├── cifar10_lab/       # 재사용 가능한 데이터/학습/평가 패키지
 │   ├── config.py
 │   ├── data.py
 │   ├── environment.py
 │   ├── engine.py
 │   ├── checkpoints.py
 │   ├── cli.py
 │   ├── paths.py
 │   ├── registry.py
 │   └── visualization.py
 ├── foundations.ipynb  # 퍼셉트론→MLP→AlexNet 교과서형 실습
 ├── cnn_internals.ipynb # AlexNet 특징맵·pooling·수용영역 실습
 ├── main.ipynb         # 메인 실행 노트북
 ├── docs/               # 설치 없이 사용하는 정적 웹 학습 자료
 ├── bootstrap.py       # 가상환경·의존성·데이터셋 자동 준비
 ├── pyproject.toml     # 패키지와 의존성 정의
 └── utils.py           # 이전 노트북 호환용 import 모듈
```

## 지원되는 백본 모델

### 신경망 기초
*   단층 퍼셉트론 (`perceptron`): 모든 픽셀에 하나의 선형 변환을 적용하는 기준선
*   다층 퍼셉트론 (`mlp`, `mlp_deep`): 은닉층과 ReLU로 비선형 표현을 학습하는 기준선

### CNN 계열
*   AlexNet (`alexnet`, `alexnet_bn`)
*   VGG (VGG11, VGG11_bn, VGG13, VGG13_bn, VGG16, VGG16_bn, VGG19, VGG19_bn)
*   ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt, Wide ResNet)
*   MobileNet (MobileNetV2)
*   EfficientNet (EfficientNetB0-B7)
*   Inception (InceptionV3)
*   ConvNeXt (ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge)
*   SENet (SE-ResNet18, SE-ResNet34, SE-ResNet50, SE-ResNet101, SE-ResNet152)

### Transformer 계열
*   Vision Transformer (ViT-Tiny, ViT-Small, ViT-Base, ViT-Large)
*   Pyramid Vision Transformer (PVT-Tiny, PVT-Small, PVT-Medium, PVT-Large)

> `main.ipynb`의 기본 비교 목록은 32×32 입력을 직접 처리할 수 있는 모델만 포함합니다. PVT는 `img_size=32`로 생성하며, 원래 큰 입력을 전제로 하는 InceptionV3는 CIFAR-10용 구조를 별도로 조정한 뒤 비교하는 것을 권장합니다.

## 권장 학습 순서

1. `foundations.ipynb`: 단일 뉴런의 가중합, 활성화 함수, AND 학습 과정을 시각적으로 확인합니다.
2. 같은 노트북에서 XOR을 통해 단일 퍼셉트론의 한계와 MLP 은닉층의 역할을 확인합니다.
3. `cnn_internals.ipynb`에서 AlexNet의 특징맵, pooling, 수용영역을 층별로 확인합니다.
4. CIFAR-10에서 `perceptron → mlp → alexnet → resnet18`을 순서대로 실행합니다.
5. `main.ipynb`에서 현대 CNN과 Transformer 백본을 같은 조건으로 비교합니다.

## 빠른 시작

Python 3.10 이상과 Git만 있으면 됩니다. `bootstrap.py`는 `.venv` 생성, 노트북 포함 의존성 설치, 실행 환경 검사, CIFAR-10 train/test 다운로드를 순서대로 수행합니다.

### Windows PowerShell

```powershell
git clone https://github.com/tember40/Backbone_tester_CNN_Transformer.git
cd Backbone_tester_CNN_Transformer
py -3 bootstrap.py
.venv\Scripts\Activate.ps1
cifar10-lab train --model resnet18 --quick
.venv\Scripts\python -m jupyter lab
```

### macOS / Linux

```bash
git clone https://github.com/tember40/Backbone_tester_CNN_Transformer.git
cd Backbone_tester_CNN_Transformer
python3 bootstrap.py
source .venv/bin/activate
cifar10-lab train --model resnet18 --quick
python -m jupyter lab
```

Jupyter Lab이 열리면 `foundations.ipynb` → `cnn_internals.ipynb` → `main.ipynb` 순서로 실행하는 것을 권장합니다.

`bootstrap.py` 실행 중에 PyTorch와 CIFAR-10을 받으므로 인터넷 연결과 수백 MB 이상의 저장 공간이 필요합니다. 노트북이 필요 없으면 `python bootstrap.py --minimal`, 데이터는 나중에 받으려면 `python bootstrap.py --skip-data`를 사용합니다.

특정 CUDA 또는 ROCm용 PyTorch가 필요한 경우에는 운영체제와 가속기 버전에 맞는 PyTorch를 먼저 설치한 뒤 이 프로젝트를 설치합니다. 실행 시 CUDA, Apple MPS, CPU 순서로 자동 감지합니다. Windows에서는 노트북의 multiprocessing 멈춤을 피하기 위해 DataLoader worker 기본값을 0으로 설정합니다.

## 설치 구성

| 목적 | 설치 명령 | 포함 내용 |
|---|---|---|
| 최소 실행 | `pip install .` | PyTorch, torchvision, 핵심 모듈 |
| 시각화 | `pip install ".[visualization]"` | matplotlib, torchinfo |
| Transformer | `pip install ".[transformers]"` | einops, timm |
| 전체 노트북 실습 | `pip install ".[notebook]"` | 시각화, Transformer, Jupyter |
| 개발·테스트 | `pip install -e ".[notebook,dev]"` | 전체 실습, pytest, ruff, build, 소스 수정 즉시 반영 |

동일한 구성은 `requirements-core.txt`, `requirements.txt`, `requirements-dev.txt`로도 설치할 수 있습니다.

설치 확인은 다음 명령으로 할 수 있습니다.

```bash
cifar10-lab doctor
```

## 사용 방법

1.  **`main.ipynb` 실행**: Jupyter에서 `main.ipynb` 파일을 열고 모든 셀을 순서대로 실행합니다.

    *   **첫 실행**: `MODEL_ID = "resnet18"`, `QUICK_RUN = True`를 그대로 사용하면 2,048개의 학습 샘플과 1 epoch로 전체 흐름을 빠르게 확인할 수 있습니다.

    *   **백본 모델 선택**: `MODEL_ID`만 원하는 백본 ID로 변경합니다. Registry가 모델 ID를 실제 생성 함수와 연결하고 입력 크기 및 추가 의존성을 관리합니다. 모델 ID는 체크포인트 파일명에도 사용되므로 VGG16/VGG19 또는 ResNet18/ResNet50의 가중치가 서로 충돌하지 않습니다.

    *   **전체 학습**: 빠른 실행이 정상적으로 끝난 다음 `QUICK_RUN = False`로 변경합니다. 빠른 실행과 전체 학습의 체크포인트는 사용자 데이터 폴더의 `checkpoints/quick`, `checkpoints/full`에 분리됩니다.

    *   **모델 구조 및 파라미터 확인**: `torchinfo.summary`를 사용하여 선택된 모델의 구조와 파라미터 수를 확인할 수 있습니다.

    *   **데이터셋 다운로드 및 학습/평가**: `main.ipynb`를 실행하면 CIFAR-10 데이터셋이 사용자 데이터 폴더에 자동 다운로드됩니다. 단, 개발 중인 저장소에 기존 `Cifar-10 dataset/`이 있으면 그 데이터를 재사용합니다. 공식 학습 데이터는 train/validation으로 재현 가능하게 분리하며, test 데이터는 최종 평가에서만 사용합니다.

        *   모델, seed, 학습률, 데이터 설정을 해시한 실험 ID로 체크포인트를 분리하여 서로 다른 실험이 덮어쓰지 않습니다.
        *   체크포인트에는 모델 ID, 가중치, optimizer 상태, 최고 validation 정확도와 학습 이력이 함께 저장됩니다.
        *   데이터 분할 및 샘플, train/validation 학습 곡선, test confusion matrix와 클래스별 정확도를 단계별로 시각화합니다.

## 명령행에서 실행

노트북 없이도 동일한 Registry, 설정, 데이터 분할과 체크포인트 형식을 사용합니다.

```bash
# 설치 및 장치 확인
cifar10-lab doctor

# 사용 가능한 모델 확인
cifar10-lab list-models

# CIFAR-10 train/test를 미리 다운로드
cifar10-lab download-data

# 적은 데이터와 1 epoch로 전체 흐름 확인
cifar10-lab train --model resnet18 --quick

# 전체 데이터로 10 epoch 학습
cifar10-lab train --model resnet18

# 저장된 빠른 실행 체크포인트 평가
cifar10-lab evaluate --model resnet18 --quick

# 1 epoch 체크포인트를 총 5 epoch까지 이어서 학습
cifar10-lab train --model resnet18 --quick --epochs 5 --resume

# 같은 조건으로 여러 모델을 학습하고 CSV·JSON·PNG 비교 보고서 생성
cifar10-lab compare --models perceptron mlp alexnet resnet18 --quick

# 각 계열의 대표 모델 입출력 형상 검증
cifar10-lab validate-models --models alexnet resnet18 vit_tiny

# 메모리와 시간이 충분한 환경에서 CIFAR-10-ready 전체 모델 검증
cifar10-lab validate-models --all

# AlexNet의 특징맵·pooling·수용영역 그림 저장
cifar10-lab visualize-cnn --model alexnet
```

`train`이나 노트북을 먼저 실행해도 데이터가 없으면 CIFAR-10을 자동으로 다운로드합니다. 따라서 `download-data`는 데이터를 미리 준비하고 싶을 때만 사용하면 됩니다. 원하는 폴더에 받으려면 `cifar10-lab download-data --data-dir ./data`를 사용합니다.

`python -m cifar10_lab`, `python -m cifar10_lab.train`, `python -m cifar10_lab.evaluate` 방식도 사용할 수 있습니다. 같은 실험 ID의 체크포인트가 있으면 학습을 건너뛰며, 저장된 optimizer와 마지막 가중치에서 이어서 학습하려면 `--resume`, 처음부터 다시 학습하려면 `--retrain`을 사용합니다. `--save-plots`를 추가하면 학습 곡선과 confusion matrix를 PNG로 저장합니다.

## 데이터와 결과 저장 위치

실행 위치와 관계없이 운영체제의 사용자 데이터 폴더 아래에 저장합니다.

- Windows: `%LOCALAPPDATA%/cifar10-backbone-lab`
- macOS: `~/Library/Application Support/cifar10-backbone-lab`
- Linux: `$XDG_DATA_HOME/cifar10-backbone-lab` 또는 `~/.local/share/cifar10-backbone-lab`

그 아래에 `data`, `checkpoints`, `results`가 생성됩니다. 저장 위치를 직접 지정하려면 `CIFAR10_LAB_HOME` 환경 변수 또는 `--data-dir`, `--checkpoint-dir`, `--results-dir` 옵션을 사용합니다. Editable 개발 환경에서는 기존 `Cifar-10 dataset` 폴더가 발견되면 다시 다운로드하지 않고 재사용합니다.

## Python 모듈에서 사용

```python
from cifar10_lab import (
    DataConfig,
    ExperimentConfig,
    TrainConfig,
    create_model,
    format_model_catalog,
)

config = ExperimentConfig(
    model_id="resnet18",
    data=DataConfig(batch_size=64, val_ratio=0.1, seed=42),
    train=TrainConfig(epochs=10, learning_rate=0.001),
)

print(format_model_catalog(cifar10_ready_only=True))
model = create_model(
    config.model_id,
    num_classes=config.num_classes,
    image_size=config.image_size,
)
```

Registry는 모델 모듈을 실제 생성 시점에만 불러옵니다. 따라서 최소 설치에서는 CNN을 사용할 수 있고, ViT/PVT를 선택할 때만 `transformers` 설치 구성이 필요합니다. InceptionV3는 목록에 참고용으로 남겨 두되 현재 32×32 파이프라인에서는 생성이 차단됩니다.

개발 환경에서는 다음 명령으로 Registry와 설정 객체의 기본 동작을 검사할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

## 개발 환경

*   Python 3.10+
*   PyTorch
*   torchvision
*   torchinfo
*   timm

## 기여

이 프로젝트에 기여하고 싶으시면 언제든지 Pull Request를 보내주세요.
