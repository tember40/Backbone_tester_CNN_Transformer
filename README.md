# CIFAR-10 Multiclass Classification Framework

이 프로젝트는 CIFAR-10 데이터셋을 활용하여 다양한 CNN 및 Transformer 기반 백본 모델들을 사용하여 Multiclass Classification을 연습할 수 있는 프레임워크를 제공합니다.

## 프로젝트 구조

```
classification
 ├── Cifar-10 dataset/  # CIFAR-10 데이터셋이 다운로드될 위치
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
 │   ├── registry.py
 │   └── visualization.py
 ├── weight/            # 학습된 모델의 가중치 파일이 저장될 위치
 │   ├── Cifar-10_resnet18.pth
 │   └── Cifar-10_convnext_tiny.pth
 │   └── ...
 ├── main.ipynb         # 메인 실행 노트북
 ├── pyproject.toml     # 패키지와 의존성 정의
 └── utils.py           # 이전 노트북 호환용 import 모듈
```

## 지원되는 백본 모델

### CNN 계열
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

## 빠른 시작

Python 3.10 이상을 권장합니다. 프로젝트 폴더에서 가상환경을 만들고 노트북용 의존성을 설치합니다.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[notebook]"
.venv\Scripts\python -m jupyter lab
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[notebook]"
.venv/bin/python -m jupyter lab
```

특정 CUDA 또는 ROCm용 PyTorch가 필요한 경우에는 운영체제와 가속기 버전에 맞는 PyTorch를 먼저 설치한 뒤 이 프로젝트를 설치합니다. 실행 시 CUDA, Apple MPS, CPU 순서로 자동 감지합니다. Windows에서는 노트북의 multiprocessing 멈춤을 피하기 위해 DataLoader worker 기본값을 0으로 설정합니다.

## 설치 구성

| 목적 | 설치 명령 | 포함 내용 |
|---|---|---|
| 최소 실행 | `pip install -e .` | PyTorch, torchvision, 핵심 모듈 |
| 시각화 | `pip install -e ".[visualization]"` | matplotlib, torchinfo |
| Transformer | `pip install -e ".[transformers]"` | einops, timm |
| 전체 노트북 실습 | `pip install -e ".[notebook]"` | 시각화, Transformer, Jupyter |
| 개발·테스트 | `pip install -e ".[notebook,dev]"` | 전체 실습, pytest, ruff, build |

동일한 구성은 `requirements-core.txt`, `requirements.txt`, `requirements-dev.txt`로도 설치할 수 있습니다.

## 사용 방법

1.  **`main.ipynb` 실행**: Jupyter에서 `main.ipynb` 파일을 열고 모든 셀을 순서대로 실행합니다.

    *   **첫 실행**: `MODEL_ID = "resnet18"`, `QUICK_RUN = True`를 그대로 사용하면 2,048개의 학습 샘플과 1 epoch로 전체 흐름을 빠르게 확인할 수 있습니다.

    *   **백본 모델 선택**: `MODEL_ID`만 원하는 백본 ID로 변경합니다. Registry가 모델 ID를 실제 생성 함수와 연결하고 입력 크기 및 추가 의존성을 관리합니다. 모델 ID는 체크포인트 파일명에도 사용되므로 VGG16/VGG19 또는 ResNet18/ResNet50의 가중치가 서로 충돌하지 않습니다.

    *   **전체 학습**: 빠른 실행이 정상적으로 끝난 다음 `QUICK_RUN = False`로 변경합니다. 빠른 실행과 전체 학습의 체크포인트는 각각 `weight/quick`, `weight/full`에 분리됩니다.

    *   **모델 구조 및 파라미터 확인**: `torchinfo.summary`를 사용하여 선택된 모델의 구조와 파라미터 수를 확인할 수 있습니다.

    *   **데이터셋 다운로드 및 학습/평가**: `main.ipynb`를 실행하면 CIFAR-10 데이터셋이 자동으로 `Cifar-10 dataset/` 디렉토리에 다운로드됩니다. 공식 학습 데이터는 train/validation으로 재현 가능하게 분리하며, test 데이터는 최종 평가에서만 사용합니다.

        *   `weight/` 디렉토리에 선택된 모델의 체크포인트(`Cifar-10_{model_id}.pth`)가 없으면 학습을 시작하고, validation 정확도가 가장 높은 모델을 저장합니다.
        *   체크포인트에는 모델 ID, 가중치, optimizer 상태, 최고 validation 정확도와 학습 이력이 함께 저장됩니다.
        *   데이터 분할 및 샘플, train/validation 학습 곡선, test confusion matrix와 클래스별 정확도를 단계별로 시각화합니다.

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
