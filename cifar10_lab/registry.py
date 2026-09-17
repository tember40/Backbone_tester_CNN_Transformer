from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    module: str
    factory: str
    family: str
    default_image_size: int = 32
    image_size_parameter: str | None = None
    dependency_group: str = "core"
    cifar10_ready: bool = True
    description: str = ""
    factory_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "factory_kwargs", MappingProxyType(dict(self.factory_kwargs)))

    def create(self, num_classes=10, image_size=None, allow_unsupported=False, **kwargs):
        selected_size = self.default_image_size if image_size is None else image_size
        if not self.cifar10_ready and not allow_unsupported:
            raise ValueError(
                f"{self.model_id!r} is listed for reference but is not ready for the "
                "32x32 CIFAR-10 pipeline. Pass allow_unsupported=True only when its "
                "input preprocessing has also been adapted."
            )

        try:
            module = import_module(self.module)
        except ModuleNotFoundError as error:
            if self.dependency_group != "core":
                raise RuntimeError(
                    f"{self.model_id!r} requires the {self.dependency_group!r} dependency "
                    f"group. Install with: pip install -e '.[{self.dependency_group}]'"
                ) from error
            raise

        factory = getattr(module, self.factory)
        factory_kwargs = dict(self.factory_kwargs)
        factory_kwargs.update(kwargs)
        factory_kwargs["num_classes"] = num_classes
        if self.image_size_parameter is not None:
            factory_kwargs[self.image_size_parameter] = selected_size
        return factory(**factory_kwargs)


class ModelRegistry:
    def __init__(self):
        self._specs = {}

    def register(self, spec):
        if spec.model_id in self._specs:
            raise ValueError(f"Model ID {spec.model_id!r} is already registered.")
        self._specs[spec.model_id] = spec

    def get(self, model_id):
        try:
            return self._specs[model_id]
        except KeyError as error:
            available = ", ".join(self.model_ids(cifar10_ready_only=True))
            raise KeyError(f"Unknown model_id {model_id!r}. Available models: {available}") from error

    def create(self, model_id, num_classes=10, image_size=None, **kwargs):
        return self.get(model_id).create(
            num_classes=num_classes,
            image_size=image_size,
            **kwargs,
        )

    def list(self, cifar10_ready_only=False):
        specs = self._specs.values()
        if cifar10_ready_only:
            specs = (spec for spec in specs if spec.cifar10_ready)
        return tuple(sorted(specs, key=lambda spec: (spec.family, spec.model_id)))

    def model_ids(self, cifar10_ready_only=False):
        return tuple(spec.model_id for spec in self.list(cifar10_ready_only))


MODEL_REGISTRY = ModelRegistry()


def _register_family(module, family, model_ids, dependency_group="core"):
    for model_id in model_ids:
        MODEL_REGISTRY.register(ModelSpec(
            model_id=model_id,
            module=module,
            factory=model_id,
            family=family,
            dependency_group=dependency_group,
        ))


_register_family("backbone.AlexNet", "AlexNet", ("alexnet", "alexnet_bn"))
_register_family(
    "backbone.VGG",
    "VGG",
    ("vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"),
)
_register_family(
    "backbone.ResNet",
    "ResNet",
    (
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d",
        "wide_resnet50_2", "wide_resnet101_2",
    ),
)
_register_family("backbone.MobileNet", "MobileNet", ("mobilenet_v2",))
_register_family(
    "backbone.EfficientNet",
    "EfficientNet",
    tuple(f"efficientnet_b{index}" for index in range(8)),
)
_register_family(
    "backbone.ConvNeXt",
    "ConvNeXt",
    ("convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xlarge"),
)
_register_family(
    "backbone.SeNet",
    "SENet",
    ("se_resnet18", "se_resnet34", "se_resnet50", "se_resnet101", "se_resnet152"),
)

for model_id in ("vit_tiny", "vit_small", "vit_base", "vit_large"):
    MODEL_REGISTRY.register(ModelSpec(
        model_id=model_id,
        module="backbone.ViT",
        factory=model_id,
        family="Vision Transformer",
        image_size_parameter="image_size",
        dependency_group="transformers",
        factory_kwargs={"patch_size": 4},
    ))

for model_id in ("pvt_tiny", "pvt_small", "pvt_medium", "pvt_large"):
    MODEL_REGISTRY.register(ModelSpec(
        model_id=model_id,
        module="backbone.PVT",
        factory=model_id,
        family="Pyramid Vision Transformer",
        image_size_parameter="img_size",
        dependency_group="transformers",
    ))

MODEL_REGISTRY.register(ModelSpec(
    model_id="inception_v3",
    module="backbone.InceptionV3",
    factory="inception_v3",
    family="Inception",
    default_image_size=299,
    cifar10_ready=False,
    description="Original stem and reductions are not compatible with the 32x32 data pipeline.",
    factory_kwargs={"aux_logits": False},
))


def create_model(model_id, num_classes=10, image_size=32, **kwargs):
    return MODEL_REGISTRY.create(
        model_id,
        num_classes=num_classes,
        image_size=image_size,
        **kwargs,
    )


def list_models(cifar10_ready_only=False):
    return MODEL_REGISTRY.list(cifar10_ready_only)


def format_model_catalog(cifar10_ready_only=False):
    specs = list_models(cifar10_ready_only)
    rows = [(spec.model_id, spec.family, spec.dependency_group) for spec in specs]
    headers = ("MODEL_ID", "FAMILY", "INSTALL GROUP")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in rows
    ]
    return "\n".join((header, divider, *body))
