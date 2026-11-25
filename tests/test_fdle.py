"""Exercise the fdle workflows used in the sampling and RNN labs."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import fdle


@pytest.fixture(scope="module", autouse=True)
def use_one_torch_thread():
    """Keep the tiny CPU workloads independent of the machine's core count."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_fit_mlp_with_tensor_pair_and_dataset():
    fdle.set_seed(4)
    signs = torch.tensor([-1.0, 1.0]).repeat(40)
    inputs = torch.stack((signs, signs * 2), dim=1)
    targets = (signs > 0).long()
    model = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 2))
    model[0].bias.requires_grad_(False)

    history = fdle.fit(
        model, (inputs, targets), TensorDataset(inputs, targets),
        epochs=4, lr=0.1, device="cpu", verbose=False,
    )

    assert set(history) == {"train_loss", "val_loss", "val_acc", "time_s",
                            "params", "best_val_acc"}
    assert all(len(history[key]) == 4 for key in ("train_loss", "val_loss", "val_acc"))
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_loss"][-1] < history["val_loss"][0]
    assert history["best_val_acc"] == max(history["val_acc"]) == 1.0
    assert history["params"] == fdle.count_params(model) == 26
    assert fdle.count_params(model, trainable_only=False) == 32
    assert history["time_s"] >= 0
    assert isinstance(fdle.get_device(), torch.device)

    scores = fdle.evaluate(model, (inputs, targets), device="cpu")
    assert set(scores) == {"loss", "acc", "y_true", "y_pred", "y_prob"}
    assert scores["acc"] == 1.0
    assert scores["loss"] == pytest.approx(history["val_loss"][-1])
    np.testing.assert_array_equal(scores["y_true"], targets.numpy())
    np.testing.assert_array_equal(scores["y_pred"], scores["y_prob"].argmax(axis=1))
    np.testing.assert_allclose(scores["y_prob"].sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_weights_examples_equally_in_uneven_batches():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, -3.0]])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)

    scores = fdle.evaluate(nn.Identity(), loader, device="cpu")

    assert scores["loss"] == pytest.approx(nn.functional.cross_entropy(logits, targets).item())
    assert scores["acc"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("binary", [False, True])
def test_evaluate_can_skip_probabilities(binary, monkeypatch):
    logits = torch.tensor([-3.0, 2.0, 0.0])
    targets = torch.tensor([0, 1, 0])
    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    inputs = logits if binary else torch.stack((-logits, logits), dim=1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    full = fdle.evaluate(nn.Identity(), loader, loss_fn=loss_fn, device="cpu")

    def unexpected_probabilities(*args, **kwargs):
        pytest.fail("Probability conversion ran with keep_outputs=False")

    monkeypatch.setattr(torch.Tensor, "sigmoid" if binary else "softmax",
                        unexpected_probabilities)
    compact = fdle.evaluate(nn.Identity(), loader, loss_fn=loss_fn, device="cpu",
                            keep_outputs=False)
    assert compact["y_prob"] is None
    assert compact["loss"] == full["loss"]
    assert compact["acc"] == full["acc"]
    np.testing.assert_array_equal(compact["y_true"], full["y_true"])
    np.testing.assert_array_equal(compact["y_pred"], full["y_pred"])


def test_fit_advances_scheduler_once_after_each_epoch():
    model = nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    batch_lrs = []

    def step(model, batch, loss_fn):
        batch_lrs.append(optimizer.param_groups[0]["lr"])
        inputs, targets = batch
        logits = model(inputs)
        return loss_fn(logits, targets), logits, targets

    loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.arange(4) % 2), batch_size=2)
    fdle.fit(model, loader, epochs=3, optimizer=optimizer, scheduler=scheduler,
             step_fn=step, device="cpu", verbose=False)

    assert batch_lrs == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025, 0.025])
    assert scheduler.get_last_lr() == pytest.approx([0.0125])


def test_fit_cnn_applies_batch_transform_during_training_and_validation():
    fdle.set_seed(5)
    inputs = torch.randn(10, 16)
    targets = torch.arange(10) % 2
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    model = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3), nn.ReLU(),
                          nn.Flatten(), nn.Linear(8, 2))
    calls = []

    def to_images(batch):
        calls.append((model.training, len(batch), batch.device.type))
        return batch.reshape(-1, 1, 4, 4)

    history = fdle.fit(
        model, loader, loader, epochs=1, batch_transform=to_images,
        device="cpu", verbose=False,
    )

    assert calls == [(mode, size, "cpu") for mode in (True, False) for size in (4, 4, 2)]
    assert np.isfinite(history["train_loss"][0])
    assert 0 <= history["val_acc"][0] <= 1


def test_fit_rnn_clips_gradients_before_optimizer_step(monkeypatch):
    fdle.set_seed(6)

    class TextRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.rnn = nn.RNN(4, 5, batch_first=True)
            self.head = nn.Linear(5, 2)

        def forward(self, tokens):
            sequence, _ = self.rnn(self.embedding(tokens))
            return self.head(sequence[:, -1])

    model = TextRNN()
    inputs = torch.randint(0, 8, (6, 4))
    loader = DataLoader(TensorDataset(inputs, torch.arange(6) % 2), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    actual_step = optimizer.step
    grad_norms = []
    max_norm = 0.01

    def check_step(*args, **kwargs):
        norm = torch.linalg.vector_norm(
            torch.cat([p.grad.flatten() for p in model.parameters()])
        ).item()
        grad_norms.append(norm)
        return actual_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", check_step)
    history = fdle.fit(model, loader, epochs=2, optimizer=optimizer,
                       grad_clip=max_norm, amp=True, device="cpu", verbose=False)

    assert grad_norms == pytest.approx([max_norm] * 6, abs=1e-6)
    assert all(np.isfinite(history["train_loss"]))
    assert history["val_loss"] == history["val_acc"] == []
    assert np.isnan(history["best_val_acc"])


def test_custom_step_accepts_three_tensor_batches():
    fdle.set_seed(7)
    left = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    right = torch.ones_like(left)
    targets = torch.arange(6) % 2
    loader = DataLoader(TensorDataset(left, right, targets), batch_size=2)
    model = nn.Linear(2, 2)
    calls = []

    def step(model, batch, loss_fn):
        calls.append((model.training, torch.is_grad_enabled()))
        left, right, labels = (part.to(model.weight.device) for part in batch)
        logits = model(torch.cat([left, right], dim=1))
        return loss_fn(logits, labels), logits, labels

    fdle.fit(model, loader, loader, epochs=1, step_fn=step, device="cpu", verbose=False)
    scores = fdle.evaluate(model, loader, step_fn=step, device="cpu")

    assert calls == [(True, True)] * 3 + [(False, False)] * 6
    np.testing.assert_array_equal(scores["y_true"], targets.numpy())
    with torch.no_grad():
        expected = model(torch.cat([left, right], dim=1)).softmax(dim=1).numpy()
    np.testing.assert_allclose(scores["y_prob"], expected, atol=1e-6)


@pytest.mark.parametrize("column_logits", [False, True])
def test_fit_and_evaluate_binary_with_bce(column_logits):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    if not column_logits:
        model = nn.Sequential(model, SqueezeLast())
    inputs = torch.tensor([[-2.0], [1.0], [2.0], [-1.0]])
    targets = torch.tensor([[0], [1], [1], [0]])
    loss_fn = nn.BCEWithLogitsLoss()

    history = fdle.fit(model, (inputs, targets), (inputs, targets), epochs=1,
                       lr=0, loss_fn=loss_fn, device="cpu", verbose=False)
    scores = fdle.evaluate(model, (inputs, targets), loss_fn=loss_fn, device="cpu")

    assert history["best_val_acc"] == scores["acc"] == 1.0
    np.testing.assert_array_equal(scores["y_pred"], targets.numpy().ravel())
    np.testing.assert_allclose(scores["y_prob"], inputs.sigmoid().numpy().ravel())


class SqueezeLast(nn.Module):
    def forward(self, inputs):
        return inputs.squeeze(-1)


@pytest.mark.parametrize("with_labels", [False, True])
def test_cross_validate_preprocesses_only_on_training_fold_and_restores_oof_order(with_labels):
    seed, k = 37, 3
    inputs = np.arange(12, dtype=float).reshape(-1, 1)
    labels = np.arange(12) % 2
    calls = []
    models = []

    def center(X_train, X_val):
        train, val = np.asarray(X_train), np.asarray(X_val)
        calls.append((train.copy(), val.copy()))
        mean = train.mean(axis=0)
        return train - mean, val - mean

    def supervised_center(X_train, y_train, X_val):
        assert isinstance(X_train, torch.Tensor)
        np.testing.assert_array_equal(y_train.numpy(), labels[X_train[:, 0].long().numpy()])
        assert len(y_train) < len(labels)
        return center(X_train, X_val)

    def model_fn():
        model = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.0], [1.0]]))
        models.append(model)
        return model

    result = fdle.cross_validate(
        model_fn,
        torch.from_numpy(inputs) if with_labels else inputs,
        torch.from_numpy(labels) if with_labels else labels,
        k=k, seed=seed, preprocess_fn=supervised_center if with_labels else center,
        epochs=1, lr=0, batch_size=3, device="cpu", verbose=False,
    )

    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    expected_prob = np.empty(len(labels))
    expected_accs = []
    assert len(calls) == len({id(model) for model in models}) == k
    for (train_idx, val_idx), (train, val) in zip(splits.split(inputs, labels), calls):
        np.testing.assert_array_equal(train, inputs[train_idx])
        np.testing.assert_array_equal(val, inputs[val_idx])
        assert not np.intersect1d(train, val).size
        assert train.mean() != pytest.approx(inputs.mean())
        expected_prob[val_idx] = 1 / (1 + np.exp(-(val[:, 0] - train.mean())))
        expected_accs.append(np.mean((expected_prob[val_idx] > 0.5) == labels[val_idx]))

    np.testing.assert_array_equal(result["oof_true"], labels)
    np.testing.assert_array_equal(result["oof_pred"], expected_prob > 0.5)
    np.testing.assert_allclose(result["oof_prob"][:, 1], expected_prob, atol=1e-6)
    np.testing.assert_allclose(result["oof_prob"].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(result["fold_accs"], expected_accs)
    assert result["mean_acc"] == pytest.approx(np.mean(expected_accs))
    assert result["std_acc"] == pytest.approx(np.std(expected_accs))
    assert result["time_s"] >= 0


def test_cross_validate_plain_kfold_without_preprocessing():
    inputs = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    labels = torch.zeros(8, dtype=torch.long)
    result = fdle.cross_validate(lambda: nn.Linear(1, 2), inputs, labels, k=2,
                                 stratified=False, epochs=1, device="cpu", verbose=False)
    assert len(result["fold_accs"]) == 2
    np.testing.assert_array_equal(result["oof_true"], labels.numpy())
    assert result["oof_prob"].shape == (8, 2)


@pytest.mark.parametrize("kind", ["numeric", "text", "ragged"])
def test_cross_validate_handles_lists_and_series(kind):
    labels = [0, 1] * 4
    preprocess = None
    inputs = [[i] for i in range(8)]
    if kind == "text":
        pd = pytest.importorskip("pandas")
        labels = pd.Series(labels, index=np.arange(10, 18))
        inputs = ["x" * i for i in range(1, 9)]
    if kind == "ragged":
        inputs = [[1] * i for i in range(1, 9)]
    if kind != "numeric":
        def preprocess(X_train, y_train, X_val):
            assert X_train.dtype == X_val.dtype == object
            assert isinstance(y_train, np.ndarray)
            return [[float(len(row))] for row in X_train], [[float(len(row))] for row in X_val]

    result = fdle.cross_validate(
        lambda: nn.Linear(1, 2), inputs, labels, preprocess_fn=preprocess,
        k=2, epochs=1, device="cpu", verbose=False,
    )
    np.testing.assert_array_equal(result["oof_true"], np.asarray(labels))
    assert result["oof_prob"].shape == (8, 2)


def test_cross_validate_passes_custom_loss_and_step_to_both_phases():
    loss_fn = nn.BCEWithLogitsLoss()
    phases = []

    def step(model, batch, loss):
        assert loss is loss_fn
        phases.append(model.training)
        inputs, targets = batch
        logits = model(inputs).squeeze(-1)
        return loss(logits, targets.float()), logits, targets

    result = fdle.cross_validate(
        lambda: nn.Linear(1, 1), np.arange(8.0).reshape(-1, 1), [0, 1] * 4,
        k=2, epochs=1, loss_fn=loss_fn, step_fn=step, device="cpu", verbose=False,
    )
    assert phases == [True, False] * 2
    assert result["oof_prob"].shape == (8,)
    np.testing.assert_array_equal(result["oof_pred"], result["oof_prob"] >= 0.5)


def test_bootstrap_default_is_seeded_paired_and_uses_percentiles():
    true = np.array([0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0])
    result = fdle.bootstrap_ci(true, pred, n_boot=80, ci=0.8, seed=8)

    rng = np.random.default_rng(8)
    indices = rng.integers(0, len(true), size=(80, len(true)))
    expected = (true[indices] == pred[indices]).mean(axis=1)
    np.testing.assert_array_equal(result["samples"], expected)
    assert result["point"] == 0.75
    assert result["mean"] == pytest.approx(expected.mean())
    assert [result["low"], result["high"]] == pytest.approx(np.quantile(expected, [0.1, 0.9]))


def test_bootstrap_custom_metric_and_invalid_inputs():
    true = np.arange(5)
    result = fdle.bootstrap_ci(true, true + 2, metric_fn=lambda a, b: (b - a).mean())
    np.testing.assert_array_equal(result["samples"], np.full(500, 2))
    assert result["point"] == result["mean"] == result["low"] == result["high"] == 2
    with pytest.raises(ValueError, match="same positive length"):
        fdle.bootstrap_ci([0], [0, 1])
    with pytest.raises(ValueError, match="confidence level"):
        fdle.bootstrap_ci([0], [0], ci=1)


def test_classification_metrics_binary_multiclass_and_unavailable_auc():
    true, pred = [0, 0, 1, 1], [0, 1, 1, 1]
    scores = fdle.classification_metrics(np.array(true)[:, None], pred, [0.1, 0.55, 0.7, 0.8])
    assert scores == pytest.approx(dict(acc=0.75, precision=2 / 3, recall=1, f1=0.8, auc=1))
    multiclass = fdle.classification_metrics([0, 1, 2], [0, 1, 2], np.eye(3))
    assert multiclass == pytest.approx(dict(acc=1, precision=1, recall=1, f1=1, auc=1))
    assert fdle.classification_metrics(true, pred)["auc"] is None
    assert fdle.classification_metrics([0, 0], [0, 0], [0.1, 0.2])["auc"] is None


def test_simple_vocab_tokenizes_counts_truncates_and_pads():
    assert fdle.tokenize("Isn't it Café 42? We’re well-known.") == [
        "isn't", "it", "café", "42", "we're", "well", "known",
    ]
    vocab = fdle.SimpleVocab(max_size=4, min_freq=2)
    assert vocab.fit(["Alpha alpha beta beta gamma gamma", "alpha rare"]) is vocab
    assert len(vocab) == 4
    assert vocab.pad_idx == 0 and vocab.unk_idx == 1
    assert vocab.encode("ALPHA beta gamma rare", max_len=3) == [2, 3, 1]
    assert vocab.encode("unknown", max_len=3) == [1, 0, 0]
    assert vocab.encode("", max_len=2) == [0, 0]

    vocab.fit(["delta delta rare"])
    assert len(vocab) == 3
    assert vocab.encode("delta alpha rare", max_len=3) == [2, 1, 1]


def test_make_text_loaders_keeps_validation_order_and_does_not_refit():
    train = ["good film", "bad film", "good ending"]
    val = ["new film", "new ending"]
    vocab = fdle.SimpleVocab(min_freq=1).fit(train)
    train_loader, val_loader = fdle.make_text_loaders(
        train, [1, 0, 1], val, np.array([0, 1]), vocab, max_len=4, batch_size=2,
    )

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert len(train_loader) == 2 and len(val_loader) == 1
    for loader, texts, expected_y in [(train_loader, train, [1, 0, 1]),
                                      (val_loader, val, [0, 1])]:
        inputs, targets = loader.dataset.tensors
        assert inputs.dtype == targets.dtype == torch.long
        assert inputs.tolist() == [vocab.encode(text, 4) for text in texts]
        assert targets.tolist() == expected_y
    assert next(iter(val_loader))[0][:, 0].tolist() == [vocab.unk_idx] * 2


def test_simple_vocab_decodes_tokens_and_can_keep_special_markers():
    vocab = fdle.SimpleVocab(min_freq=1).fit(["hello, world!"])
    ids = torch.tensor(vocab.encode("Hello out world", max_len=4))
    assert vocab.decode(ids) == "hello world"
    assert vocab.decode(ids, skip_special=False) == "hello <unk> world <pad>"
    assert vocab.decode([-1, 99], skip_special=False) == "<unk> <unk>"
    vocab.fit(["new"])
    assert vocab.decode([2]) == "new"


def test_results_table_collects_fit_results_and_displays_extra_columns(capsys):
    table = fdle.ResultsTable()
    table.show()
    assert "No results yet" in capsys.readouterr().out

    inputs, targets = torch.ones(2, 1), torch.zeros(2, dtype=torch.long)
    history = fdle.fit(nn.Linear(1, 2), (inputs, targets), (inputs, targets),
                       epochs=1, name="baseline", table=table, device="cpu", verbose=False)
    table.add("challenge", 20, 0.75, 0.2, cell="gru")

    assert table.rows[0] == dict(name="baseline", params=history["params"],
                                 val_acc=history["val_acc"][-1], time_s=history["time_s"])
    assert table.rows[1]["cell"] == "gru"
    table.show()
    output = capsys.readouterr().out
    assert "name | params | val_acc | time_s | cell" in output
    assert "baseline" in output and "challenge" in output and "gru" in output


def test_fit_records_final_validation_accuracy_in_table(monkeypatch):
    accuracies = iter([0.9, 0.5])

    def evaluate_epoch(*args, **kwargs):
        return {"loss": 1.0, "acc": next(accuracies)}

    monkeypatch.setattr(fdle.train, "evaluate", evaluate_epoch)
    data = (torch.ones(2, 1), torch.tensor([0, 1]))
    table = fdle.ResultsTable()
    history = fdle.fit(nn.Linear(1, 2), data, data, epochs=2, table=table,
                       name="baseline", device="cpu", verbose=False)
    assert history["val_acc"] == [0.9, 0.5]
    assert history["best_val_acc"] == 0.9
    assert table.rows[0]["val_acc"] == 0.5


def test_results_table_formats_text_and_escapes_html(capsys):
    table = fdle.ResultsTable()
    config = {"hidden": "large raw config"}
    table.add("<model>", 50, 0.847, 1.63, lr=np.float32(0.01234),
              config=config, details={"x": 1})
    table.add("second", 20, 0.5, 0.55)

    table.show()
    text = capsys.readouterr().out
    html = table._repr_html_()
    for output in (text, html):
        for value in ("84.7%", "1.6 s", "0.012", "[dict, size=1]"):
            assert value in output
        assert "config" not in output
    assert "<model>" in text
    assert "<table>" in html and "<td>&lt;model&gt;</td>" in html
    assert "<model>" not in html
    assert table.rows[0]["config"] is config


@pytest.mark.parametrize("obj, declaration", [
    (fdle.fit, "def fit("), (fdle.evaluate, "def evaluate("),
    (fdle.SimpleVocab, "class SimpleVocab"),
])
def test_show_source_prints_header_and_valid_python(obj, declaration, capsys):
    fdle.show_source(obj)
    source = capsys.readouterr().out
    assert source.startswith(f"# {obj.__module__}.{obj.__qualname__}\n")
    assert declaration in source
    compile(source, "<fdle source>", "exec")


def test_notebook_plots_return_figures_with_expected_data():
    history = fdle.plot_history(dict(train_loss=[1, 0.5], val_loss=[0.7, 0.6],
                                    val_acc=[0.5, 0.75]), title="Training")
    cv = fdle.plot_cv_scores([0.6, 0.8], single_split_acc=0.9)
    confusion = fdle.plot_confusion([0, 1], [0, 0], class_names=["A", "B", "C"])
    try:
        np.testing.assert_array_equal(history.axes[1].lines[0].get_ydata(), [0.5, 0.75])
        assert [bar.get_height() for bar in cv.axes[0].patches] == [0.6, 0.8]
        assert cv.axes[0].lines[1].get_ydata() == [0.9, 0.9]
        np.testing.assert_array_equal(confusion.axes[0].images[0].get_array(),
                                      [[1, 0, 0], [1, 0, 0], [0, 0, 0]])
    finally:
        for fig in (history, cv, confusion):
            plt.close(fig)
