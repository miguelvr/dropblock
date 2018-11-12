import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from torchvision.models.resnet import ResNet, Bottleneck

from dropblock import DropBlock2D


class DropBlockResNet(ResNet):

    def __init__(self, block, layers, num_classes=1000, drop_prob=0.1, block_size=7):
        super(DropBlockResNet, self).__init__(block, layers, num_classes)
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        x = self.dropblock(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    return DropBlockResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def log_validation_results(engine, evaluator, loader, pbar):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} "
        "Avg loss: {:.2f}".format(engine.state.epoch, avg_accuracy, avg_nll)
    )


def log_test_results(engine, evaluator, loader, pbar):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    pbar.log_message(
        "Test Results - Avg accuracy: {:.2f}".format(avg_accuracy)
    )


if __name__ == '__main__':
    root = './data'
    bsize = 256
    workers = 4
    epochs = 50
    lr = 0.001
    momentum = 0.9
    drop_prob = 0.1
    block_size = 5
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    NUM_TRAIN = 45000
    NUM_VAL = 5000

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True, transform=transform)

    train_set, dev_set = torch.utils.data.random_split(train_set, [NUM_TRAIN, NUM_VAL])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=bsize,
                                             shuffle=True, num_workers=workers)

    test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define model
    model = resnet50(num_classes=len(classes), drop_prob=drop_prob, block_size=block_size)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # create ignite engines
    trainer = create_supervised_trainer(model=model,
                                        optimizer=optimizer,
                                        loss_fn=criterion,
                                        device=device)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)},
                                            device=device)

    # ignite handlers
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results, evaluator, dev_loader, pbar)
    trainer.add_event_handler(Events.COMPLETED, log_test_results, evaluator, test_loader, pbar)

    # start training
    trainer.run(train_loader, max_epochs=epochs)
