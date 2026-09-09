import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


class SyntheticImageDataset(Dataset):
    def __init__(
        self,
        num_samples=256,
        num_classes=10,
        image_channels=3,
        image_size=16,
        sample_seed=0,
        template_seed=42,
    ):
        self.num_classes = num_classes

        template_generator = torch.Generator().manual_seed(template_seed)
        self.templates = torch.rand(
            num_classes,
            image_channels,
            image_size,
            image_size,
            generator=template_generator,
        )

        sample_generator = torch.Generator().manual_seed(sample_seed)
        self.targets = torch.randint(
            0,
            num_classes,
            (num_samples,),
            generator=sample_generator,
        )

        noise = torch.randn(
            num_samples,
            image_channels,
            image_size,
            image_size,
            generator=sample_generator,
        ) * 0.05

        self.data = torch.clamp(self.templates[self.targets] + noise, 0.0, 1.0)

    def __len__(self):
        return int(self.targets.numel())

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class RingTopology:
    def __init__(self, num_nodes):
        self.num_nodes = int(num_nodes)

    def get_neighbors(self, peer_id):
        peer_id = int(peer_id)
        left = (peer_id - 1) % self.num_nodes
        right = (peer_id + 1) % self.num_nodes
        return [left, peer_id, right]


class FullTopology:
    def __init__(self, num_nodes):
        self.num_nodes = int(num_nodes)

    def get_neighbors(self, peer_id):
        return list(range(self.num_nodes))


class LightweightCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Linear(32 * 2 * 2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_lightweight_cnn():
    return LightweightCNN(in_channels=3, num_classes=10)


class VirtualPeerManager:
    def __init__(
        self,
        model_fn=None,
        num_peers=8,
        device="cuda",
        lr=0.01,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.num_peers = int(num_peers)
        model_fn = model_fn or build_lightweight_cnn

        base_model = model_fn().to(self.device)

        self.models = []
        for _ in range(self.num_peers):
            model = model_fn().to(self.device)
            model.load_state_dict(base_model.state_dict())
            self.models.append(model)

        self.optimizers = [
            torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
            for model in self.models
        ]

        if self.device.startswith("cuda"):
            self.streams = [
                torch.cuda.Stream(device=self.device)
                for _ in range(self.num_peers)
            ]
        else:
            self.streams = [None for _ in range(self.num_peers)]

    def flatten_params(self, model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def unflatten_params(self, model, flat_params):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            slice_ = flat_params[offset:offset + numel].reshape_as(p)
            p.data.copy_(slice_.to(device=p.device, non_blocking=True))
            offset += numel

    def flatten_grads(self, model):
        flats = []
        for p in model.parameters():
            if p.grad is None:
                flats.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                flats.append(p.grad.detach().reshape(-1))
        return torch.cat(flats)

    def assign_flat_grads(self, model, flat_grads):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            grad_slice = flat_grads[offset:offset + numel].reshape_as(p)
            grad_slice = grad_slice.to(device=p.device, non_blocking=True)

            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)

            offset += numel


class DataPartitioner:
    def __init__(
        self,
        num_peers=8,
        batch_size=64,
        image_channels=3,
        image_size=16,
        num_classes=10,
        train_samples_per_peer=256,
        test_samples=256,
        seed=0,
    ):
        self.num_peers = int(num_peers)
        self.batch_size = int(batch_size)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.train_samples_per_peer = int(train_samples_per_peer)
        self.test_samples = int(test_samples)
        self.seed = int(seed)

    def _dataset(self, num_samples, sample_seed):
        return SyntheticImageDataset(
            num_samples=num_samples,
            num_classes=self.num_classes,
            image_channels=self.image_channels,
            image_size=self.image_size,
            sample_seed=sample_seed,
            template_seed=self.seed + 42,
        )

    def _subset_by_classes(self, dataset, classes):
        class_set = set(int(c) for c in classes)
        mask = torch.tensor([int(y.item() in class_set) for _, y in dataset], dtype=torch.bool)
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

        if len(indices) == 0:
            indices = list(range(len(dataset)))

        return Subset(dataset, indices)

    def get_peer_dataloaders(self, iid=True):
        test_dataset = self._dataset(
            num_samples=self.test_samples,
            sample_seed=self.seed + 999,
        )

        pin_memory = torch.cuda.is_available()

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        peer_loaders = []

        if iid:
            for peer_id in range(self.num_peers):
                dataset = self._dataset(
                    num_samples=self.train_samples_per_peer,
                    sample_seed=self.seed + 1000 + peer_id,
                )
                peer_loaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=pin_memory,
                    )
                )
        else:
            global_dataset = self._dataset(
                num_samples=self.train_samples_per_peer * self.num_peers,
                sample_seed=self.seed + 5000,
            )

            for peer_id in range(self.num_peers):
                classes = [
                    peer_id % self.num_classes,
                    (peer_id + 1) % self.num_classes,
                ]
                subset = self._subset_by_classes(global_dataset, classes)
                peer_loaders.append(
                    DataLoader(
                        subset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=pin_memory,
                    )
                )

        return peer_loaders, test_loader
