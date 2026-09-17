from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 64
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int | None = None
    data_root: str | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1.")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be zero or greater.")
        for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 when provided.")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "vgg19_bn"
    num_classes: int = 10
    image_size: int = 32
    device: str = "auto"
    weight_dir: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if not self.model_id:
            raise ValueError("model_id must not be empty.")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if self.image_size < 1:
            raise ValueError("image_size must be at least 1.")
        if self.weight_dir is not None and not self.weight_dir:
            raise ValueError("weight_dir must not be empty when provided.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        payload = dict(values)
        payload["data"] = DataConfig(**payload.get("data", {}))
        payload["train"] = TrainConfig(**payload.get("train", {}))
        return cls(**payload)
