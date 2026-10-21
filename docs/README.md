# Backbone Lab 웹 학습 자료

이 폴더는 별도 Python 서버 없이 동작하는 정적 웹 학습 자료입니다.

## 로컬 미리보기

저장소 루트에서 다음 명령을 실행합니다.

```powershell
# Windows · ML 패키지나 가상환경 없이도 Python만 있으면 됩니다.
py -3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

macOS/Linux에서는 `python3 -m http.server 8000 --bind 127.0.0.1 --directory docs`를 사용합니다. 이미 만든 가상환경의 Python 경로로 실행해도 됩니다. 이는 이 컴퓨터에서만 보는 로컬 미리보기이며, 외부 공개는 GitHub Pages 배포를 사용합니다.

브라우저에서 `http://localhost:8000`을 엽니다. 단순히 HTML 파일을 직접 열 수도 있지만,
브라우저 보안 정책과 실제 GitHub Pages 경로를 동일하게 확인하려면 로컬 서버 사용을 권장합니다.

## 구조

```text
docs/
├── index.html                  # 학습 자료 홈과 전체 목차
├── chapters/                   # 장별 학습 페이지
├── templates/chapter-template.html # 새 단원 작성용 전공 학습 자료 템플릿
├── assets/css/site.css         # 모든 장이 공유하는 디자인
└── assets/js/                  # 서버 없이 실행되는 장별 실험
```

새 장은 `templates/chapter-template.html`을 복사하고 `STYLE_GUIDE.md`의 구성 원칙에 맞춰 작성합니다.
오른쪽 용어장은 `assets/js/glossary.js`에서 장별 목록을 관리합니다. 새 장을 추가할 때는 본문 절 링크와 영어·한국어 용어를 함께 등록합니다.

현재 완성된 장:

- `01-perceptron.html`: 퍼셉트론의 계산과 학습 규칙
- `02-linear-separability.html`: 결정경계, 선형 분리 가능성, XOR 실험
- `03-multilayer-perceptron.html`: 은닉층, 순전파·역전파, XOR 학습 실험
- `04-convolution-filters.html`: 합성곱 계산, 출력 크기, 필터별 특징맵 실험
- `05-feature-maps-activations.html`: 특징맵 채널, 텐서 차원, ReLU 활성화 실험
- `06-pooling-receptive-field.html`: 최대·평균 풀링 계산, 출력 크기, 수용영역 누적 실험
- `07-alexnet.html`: AlexNet 논문, CIFAR-10 구조 변형, 파라미터와 Dropout 실험
- `08-vgg.html`: VGG 논문, 작은 필터와 깊이별 구조, CIFAR-10 모델과 메모리 실험
- `09-resnet.html`: 잔차 연결, 단계별 특징맵, CIFAR-10 구현과 ResNeXt·Wide ResNet 비교
- `10-senet.html`: 채널별 squeeze·excitation·scale, 축소 비율, SE-ResNet 코드 실습
- `11-mobilenetv2.html`: 깊이별·점별 합성곱, 역잔차와 선형 병목, 32×32 단계별 구조 실습
- `12-efficientnet.html`: 복합 스케일링, MBConv·SE, B0~B3의 실제 정수 구조
- `13-attention-basics.html`: Q/K/V, 안정적 softmax 수치와 Multi-Head 차원
- `14-vision-transformer.html`: 패치 펼침·투영·위치·CLS와 Encoder
- `15-pvt.html`: 특징 피라미드와 SRA, 작은 입력의 K/V 수 및 잔차
- `16-convnext.html`: 현대 CNN 블록, 채널 LayerNorm, 큰 커널과 padding
- `17-cnn-transformer.html`: 연결·공유·위치·자원 비교와 먼 입력 변화 실습
- `18-backbone-selection.html`: 조건 통제·검증 선택·합성 학습 곡선·혼동행렬·실패 가설

총 18장입니다. 장별 수치와 시각화의 검증 범위·진행 상태는 `UPGRADE_PROGRESS.md`에 기록합니다. 브라우저의 합성 예시는 실제 CIFAR-10 학습 결과와 구분합니다.

`main` 브랜치의 `docs/`·홈 리다이렉트·`chapters/` 변경은 GitHub Actions가 GitHub Pages에 자동 배포합니다. 배포 아티팩트는 `/docs/`의 실제 자료와 루트 `/`·`/chapters/`의 리다이렉트만 포함합니다. 저장소 루트를 게시하는 Pages 경로와 같은 주소 배치를 유지하며, 데이터셋·가상환경·체크포인트를 아티팩트에 넣지 않습니다.
