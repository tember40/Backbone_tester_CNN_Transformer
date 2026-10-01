(() => {
  "use strict";

  const rail = document.querySelector(".chapter-glossary[data-chapter]");
  if (!rail) return;

  const chapterFiles = {
    "01": "01-perceptron.html",
    "02": "02-linear-separability.html",
    "03": "03-multilayer-perceptron.html",
    "04": "04-convolution-filters.html",
    "05": "05-feature-maps-activations.html",
    "06": "06-pooling-receptive-field.html",
    "07": "07-alexnet.html",
    "08": "08-vgg.html",
  };
  const chapterNames = {
    "01": "퍼셉트론", "02": "선형 분리와 XOR", "03": "다층 퍼셉트론",
    "04": "합성곱과 필터", "05": "특징맵과 활성화",
    "06": "Pooling과 수용영역", "07": "AlexNet", "08": "VGG",
  };
  const term = (english, korean, meaning, chapter, section) =>
    ({english, korean, meaning, chapter, section});
  const words = {
    perceptron: term("Perceptron", "퍼셉트론", "입력의 가중합을 기준으로 두 부류를 나누는 단일 뉴런 모델입니다.", "01", "anatomy"),
    weightedSum: term("Weighted Sum", "가중합", "각 입력에 가중치를 곱해 더하고 편향을 보탠 값입니다.", "01", "anatomy"),
    weight: term("Weight", "가중치", "입력이 예측에 미치는 크기와 방향을 조정하는 학습 변수입니다.", "01", "anatomy"),
    bias: term("Bias", "편향", "가중합에 더해 결정경계의 위치를 옮기는 학습 변수입니다.", "01", "anatomy"),
    decisionBoundary: term("Decision Boundary", "결정경계", "모델의 예측이 한 부류에서 다른 부류로 바뀌는 경계입니다.", "01", "learning"),
    learningRate: term("Learning Rate", "학습률", "오차에 따라 가중치를 한 번에 얼마나 바꿀지 정하는 크기입니다.", "01", "learning"),
    linearSeparability: term("Linear Separability", "선형 분리", "직선 또는 초평면 하나로 서로 다른 부류를 나눌 수 있는 성질입니다.", "02", "geometry"),
    hyperplane: term("Hyperplane", "초평면", "고차원 공간에서 선형 결정경계가 되는 평평한 면입니다.", "02", "geometry"),
    xor: term("XOR", "배타적 논리합", "두 입력이 다를 때만 1인 논리 연산으로, 단일 직선으로 분리할 수 없습니다.", "02", "xor"),
    expressivePower: term("Expressive Power", "표현력", "모델 구조가 만들 수 있는 함수와 결정경계의 범위입니다.", "02", "bridge"),
    mlp: term("Multilayer Perceptron", "다층 퍼셉트론", "입력층과 출력층 사이에 하나 이상의 은닉층을 둔 신경망입니다.", "03", "anatomy"),
    hiddenLayer: term("Hidden Layer", "은닉층", "입력과 출력 사이에서 중간 표현을 계산하는 층입니다.", "03", "anatomy"),
    activationFunction: term("Activation Function", "활성화 함수", "층의 출력을 변환하여 여러 층이 비선형 관계를 표현하게 합니다.", "03", "forward"),
    forwardPass: term("Forward Pass", "순전파", "입력부터 출력까지 층별 계산을 차례로 수행하는 과정입니다.", "03", "forward"),
    backpropagation: term("Backpropagation", "역전파", "출력 오차의 기울기를 뒤쪽 층에서 앞쪽 층으로 전달하는 절차입니다.", "03", "learning"),
    gradient: term("Gradient", "기울기", "각 매개변수를 바꿀 때 손실이 어느 방향으로 얼마나 변하는지 나타냅니다.", "03", "learning"),
    convolution: term("Convolution", "합성곱", "작은 필터를 이미지 위로 이동하며 위치별 특징을 계산하는 연산입니다.", "04", "principles"),
    kernel: term("Kernel / Filter", "커널·필터", "이미지의 작은 영역에 적용하는, 학습 가능한 가중치 배열입니다.", "04", "calculation"),
    stride: term("Stride", "보폭", "필터나 풀링 창이 한 번에 이동하는 칸 수입니다.", "04", "geometry"),
    padding: term("Padding", "패딩", "가장자리 바깥에 값을 덧붙여 출력 크기와 경계 계산을 조정합니다.", "04", "geometry"),
    weightSharing: term("Weight Sharing", "가중치 공유", "같은 필터 가중치를 이미지의 여러 위치에서 재사용하는 규칙입니다.", "04", "principles"),
    featureMap: term("Feature Map", "특징맵", "필터가 각 공간 위치에서 계산한 응답을 배열로 모은 결과입니다.", "04", "lab"),
    tensor: term("Tensor", "텐서", "이미지와 특징을 차원별로 정리한 다차원 숫자 배열입니다.", "05", "tensor"),
    channel: term("Channel", "채널", "특징 텐서에서 서로 다른 종류의 반응을 담는 축입니다.", "05", "channels"),
    relu: term("ReLU", "정류 선형 함수", "음수 입력은 0으로, 양수 입력은 그대로 통과시키는 활성화 함수입니다.", "05", "activation"),
    sparsity: term("Sparsity", "희소성", "활성화 결과에서 0인 값이 많이 나타나는 성질입니다.", "05", "activation"),
    pooling: term("Pooling", "풀링", "작은 공간 영역의 여러 값을 하나로 요약하는 연산입니다.", "06", "pooling"),
    maxPooling: term("Max Pooling", "최대 풀링", "각 창에서 가장 큰 응답 하나를 남기는 풀링입니다.", "06", "comparison"),
    averagePooling: term("Average Pooling", "평균 풀링", "각 창의 값을 평균 내어 남기는 풀링입니다.", "06", "comparison"),
    receptiveField: term("Receptive Field", "수용영역", "특징맵의 한 칸에 영향을 줄 수 있는 원본 입력의 범위입니다.", "06", "receptive-field"),
    jump: term("Jump", "점프", "인접한 특징맵 값의 중심이 원본 입력에서 떨어진 간격입니다.", "06", "receptive-field"),
    alexnet: term("AlexNet", "알렉스넷", "대규모 이미지 분류에서 깊은 CNN의 가능성을 보여준 2012년 모델입니다.", "07", "paper"),
    imagenet: term("ImageNet", "이미지넷", "대규모 이미지 인식 연구에 사용된 이미지 데이터셋입니다.", "07", "paper"),
    lrn: term("LRN", "국소 응답 정규화", "AlexNet에서 인접 채널 반응을 정규화하는 데 사용한 기법입니다.", "07", "breakthrough"),
    dropout: term("Dropout", "드롭아웃", "학습 중 일부 뉴런 출력을 확률적으로 제외해 과적합을 줄입니다.", "07", "lab"),
    logits: term("Logits", "분류 점수", "확률로 정규화하기 전 마지막 선형층이 내놓은 클래스별 값입니다.", "07", "code"),
    vgg: term("VGG", "VGG 계열", "작은 3×3 필터를 반복하여 깊이의 효과를 비교한 CNN 계열입니다.", "08", "paper"),
    batchNorm: term("Batch Normalization", "배치 정규화", "학습 배치의 통계로 중간 출력을 정규화하고 조정하는 층입니다.", "08", "code"),
    adaptivePool: term("Adaptive Average Pooling", "적응형 평균 풀링", "입력 크기에 맞춰 구역을 정해 지정된 출력 공간 크기로 평균을 냅니다.", "08", "adaptation"),
    parameter: term("Parameter", "파라미터", "학습 과정에서 갱신되는 가중치와 편향의 원소입니다.", "01", "anatomy"),
    mib: term("MiB", "메비바이트", "2의 20제곱 바이트를 한 단위로 세는 메모리 크기입니다.", "08", "lab"),
  };
  const chapterWords = {
    "01": [["perceptron", "anatomy"], ["weightedSum", "calculator"], ["weight", "anatomy"], ["bias", "anatomy"], ["decisionBoundary", "learning"], ["learningRate", "learning"]],
    "02": [["linearSeparability", "geometry"], ["decisionBoundary", "geometry"], ["hyperplane", "geometry"], ["xor", "xor"], ["expressivePower", "bridge"]],
    "03": [["mlp", "anatomy"], ["hiddenLayer", "anatomy"], ["activationFunction", "forward"], ["forwardPass", "forward"], ["backpropagation", "learning"], ["gradient", "learning"]],
    "04": [["convolution", "principles"], ["kernel", "calculation"], ["stride", "geometry"], ["padding", "geometry"], ["weightSharing", "principles"], ["featureMap", "lab"]],
    "05": [["tensor", "tensor"], ["channel", "channels"], ["featureMap", "channels"], ["relu", "activation"], ["activationFunction", "activation"], ["sparsity", "activation"]],
    "06": [["pooling", "pooling"], ["maxPooling", "comparison"], ["averagePooling", "comparison"], ["stride", "geometry"], ["receptiveField", "receptive-field"], ["jump", "receptive-field"]],
    "07": [["alexnet", "paper"], ["imagenet", "paper"], ["relu", "breakthrough"], ["lrn", "breakthrough"], ["dropout", "lab"], ["logits", "code"]],
    "08": [["vgg", "paper"], ["kernel", "small-kernels"], ["receptiveField", "small-kernels"], ["maxPooling", "architecture"], ["batchNorm", "code"], ["adaptivePool", "adaptation"], ["parameter", "lab"], ["mib", "lab"]],
  };

  const currentChapter = rail.dataset.chapter;
  const entries = chapterWords[currentChapter];
  if (!entries) return;
  const list = rail.querySelector(".glossary-list");
  const make = (tag, className, content) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content) element.textContent = content;
    return element;
  };
  const locationLabel = (section) => {
    const nav = document.querySelector('.chapter-sidebar a[href="#' + section + '"]');
    if (!nav) return "본문에서 보기";
    const number = nav.querySelector("span")?.textContent || "";
    return number + " " + nav.textContent.replace(number, "").trim();
  };
  const details = [];

  for (const [key, section] of entries) {
    const word = words[key];
    const item = make("details", "glossary-entry");
    const summary = make("summary");
    const english = make("strong", "glossary-english", word.english);
    english.lang = "en";
    summary.append(english, make("small", "glossary-korean", word.korean));

    const note = make("div", "glossary-note");
    note.append(make("p", "", word.meaning));
    const links = make("div", "glossary-links");
    const here = make("a", "", "이 장 · " + locationLabel(section));
    here.href = "#" + section;
    links.append(here);
    if (word.chapter !== currentChapter || word.section !== section) {
      const learn = make("a", "", "함께 읽기 · " + Number(word.chapter) + "장 " + chapterNames[word.chapter]);
      learn.href = chapterFiles[word.chapter] + "#" + word.section;
      links.append(learn);
    }
    note.append(links);
    item.append(summary, note);
    item.addEventListener("toggle", () => {
      if (item.open) details.forEach((other) => { if (other !== item) other.open = false; });
    });
    details.push(item);
    list.append(item);
  }

  const toggle = make("button", "glossary-toggle", "핵심 용어");
  toggle.type = "button";
  toggle.setAttribute("aria-controls", "chapterGlossary");
  toggle.setAttribute("aria-expanded", "false");
  const backdrop = make("div", "glossary-backdrop");
  const close = make("button", "glossary-close", "닫기 ×");
  close.type = "button";
  close.setAttribute("aria-label", "용어장 닫기");
  rail.querySelector(".glossary-heading").append(close);
  document.body.append(toggle, backdrop);
  const compact = window.matchMedia("(max-width: 1040px)");

  function setOpen(open) {
    rail.inert = compact.matches && !open;
    rail.classList.toggle("is-open", open);
    backdrop.classList.toggle("is-visible", open);
    toggle.setAttribute("aria-expanded", String(open));
    if (open) rail.querySelector("summary")?.focus();
    else if (compact.matches) toggle.focus();
  }
  toggle.addEventListener("click", () => setOpen(!rail.classList.contains("is-open")));
  close.addEventListener("click", () => setOpen(false));
  backdrop.addEventListener("click", () => setOpen(false));
  rail.addEventListener("click", (event) => {
    const link = event.target.closest(".glossary-links a");
    if (compact.matches && link?.getAttribute("href")?.startsWith("#")) setOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && rail.classList.contains("is-open")) setOpen(false);
  });
  compact.addEventListener("change", () => {
    if (rail.classList.contains("is-open")) setOpen(false);
    rail.inert = compact.matches;
  });
  rail.inert = compact.matches;
})();
