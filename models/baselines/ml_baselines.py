import json

import torch

from models.baselines.features import RMS, MAV, HP, SSC, WL
from sklearn.metrics import classification_report
import numpy as np
import json as js


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


def dataset_to_list(dataset):
    samples = []
    labels = []
    for batch in dataset:
        x, y = batch[0], batch[-1]
        samples.append(np.array(x[None, ...]))
        labels.append(np.array(y[None, ...]))

    samples = np.stack(samples, axis=0)
    labels = np.stack(labels, axis=0)
    return samples.reshape(samples.shape[0], -1), labels.reshape(labels.shape[0])


def get_features(signal_windows):
    features = []
    funcs = [RMS, MAV, HP]
    for sig in signal_windows:
        tmp = []
        for func in funcs:
            tmp.append(np.apply_along_axis(func, -1, sig))
        new_ins = np.concatenate(tmp, axis=-1)
        new_ins = new_ins.flatten()
        features.append(new_ins)
    return np.array(features)


def get_baseline_performance(
        models,
        train_data,
        test_data,
        metrics=None,
        dataset=True,
        use_features=True,
):
    if dataset:
        fn = dataset_to_list
    else:
        fn = dataloader_to_list
    train_X, train_Y = fn(train_data)
    test_X, test_Y = fn(test_data)

    print("-" * 10 + "Testing performance without feature:" + "-" * 10)

    for model in models:
        print("model: ", repr(model))
        model.fit(train_X, train_Y)
        logits = model.predict_proba(test_X)
        pred = np.argmax(logits, axis=-1)
        # print(
        #     classification_report(
        #         test_Y,
        #         pred,
        #     )
        # )
        results = {}
        pt_logits = torch.from_numpy(logits)
        pt_y = torch.from_numpy(test_Y).squeeze(-1)
        for name, fn in metrics.items():
            results[f"{name}"] = float(fn(pt_logits, pt_y))
        print(json.dumps(results, indent=4))

    if use_features:
        print("-" * 10 + "Testing performance with feature:" + "-" * 10)

        train_X = get_features(train_X)
        test_X = get_features(test_X)

        for model in models:
            print("model: ", repr(model))
            model.fit(train_X, train_Y)
            logits = model.predict_proba(test_X)
            pred = np.argmax(logits, axis=-1)
            # print(
            #     classification_report(
            #         test_Y,
            #         pred,
            #     )
            # )
            results = {}
            pt_logits = torch.from_numpy(logits)
            pt_y = torch.from_numpy(test_Y).squeeze(-1)
            for name, fn in metrics.items():
                results[f"{name}"] = float(fn(pt_logits, pt_y))
            print(json.dumps(results, indent=4))
