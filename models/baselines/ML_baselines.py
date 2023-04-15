from sklearn.metrics import classification_report
import numpy as np


def dataloader_to_list(dataloader):
    samples = []
    labels = []
    for batch in dataloader.dataset:
        x, y = batch[0], batch[-1]
        samples.append(np.array(x[None, ...]))
        labels.append(np.array(y[None, ...]))

    samples = np.stack(samples, axis=0)
    labels = np.stack(labels, axis=0)
    return samples.reshape(samples.shape[0], -1), labels.reshape(labels.shape[0])


def get_baseline_performance(
        model,
        train_loader,
        test_loader,
):
    train_X, train_Y = dataloader_to_list(train_loader)
    test_X, test_Y = dataloader_to_list(test_loader)

    model.fit(train_X, train_Y)
    print(
        classification_report(
            test_Y,
            model.predict(test_X),
        )
    )
