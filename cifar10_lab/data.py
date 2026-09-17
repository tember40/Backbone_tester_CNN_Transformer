import torch
import torchvision
import torchvision.transforms as transforms

from .environment import default_num_workers
from .paths import resolve_data_dir


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_cifar10_data(
    batch_size=64,
    val_ratio=0.1,
    seed=42,
    num_workers=None,
    pin_memory=None,
    image_size=32,
    data_root=None,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
):
    """CIFAR-10을 train/validation/test로 재현 가능하게 분리한다."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if image_size < 1:
        raise ValueError("image_size must be at least 1.")
    sample_limits = {
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "max_test_samples": max_test_samples,
    }
    for name, value in sample_limits.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1 when provided.")

    resize = [] if image_size == 32 else [transforms.Resize((image_size, image_size))]
    transform_train = transforms.Compose([*resize,
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_eval = transforms.Compose([*resize,
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    resolved_data_root = resolve_data_dir(data_root)
    train_dataset_aug = torchvision.datasets.CIFAR10(
        root=resolved_data_root, train=True, download=True, transform=transform_train
    )
    train_dataset_eval = torchvision.datasets.CIFAR10(
        root=resolved_data_root, train=True, download=False, transform=transform_eval
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=resolved_data_root, train=False, download=True, transform=transform_eval
    )

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset_aug), generator=split_generator).tolist()
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    train_dataset = torch.utils.data.Subset(train_dataset_aug, train_indices)
    val_dataset = torch.utils.data.Subset(train_dataset_eval, val_indices)
    if max_test_samples is not None:
        test_generator = torch.Generator().manual_seed(seed + 1)
        test_indices = torch.randperm(
            len(test_dataset), generator=test_generator
        ).tolist()[:max_test_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    worker_count = default_num_workers(num_workers)
    use_pinned_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    loader_options = {
        "batch_size": batch_size,
        "num_workers": worker_count,
        "pin_memory": use_pinned_memory,
        "persistent_workers": worker_count > 0,
    }
    shuffle_generator = torch.Generator().manual_seed(seed)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=shuffle_generator,
        **loader_options,
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_options
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **loader_options
    )
    return trainloader, valloader, testloader, CIFAR10_CLASSES
