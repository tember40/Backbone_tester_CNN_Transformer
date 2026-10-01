# Backbone Lab 웹 학습 자료

이 폴더는 별도 Python 서버 없이 동작하는 정적 웹 학습 자료입니다.

## 로컬 미리보기

저장소 루트에서 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory docs
```

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

`main` 브랜치의 `docs/` 변경은 GitHub Actions가 GitHub Pages에 자동 배포합니다.
