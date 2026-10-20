"""Model loading, training (transformers Trainer) and batched inference for pair classifiers.

The training setup follows the "Fine-Tuning BERT-perturbed" notebook: evaluation and
checkpointing every epoch, the best epoch (by development F1) restored at the end,
early stopping after three epochs without improvement, weight decay 0.01 and gradient
accumulation. Unlike the notebook, batches are padded dynamically to their longest
example instead of to a fixed length, which makes training much faster.
"""

from __future__ import annotations

import contextlib
import inspect
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from packaging.version import Version

from .._optional import require
from .._utils.warn import warn
from ..exceptions import ConfigurationError, ModelNotAvailableError, SimchologyHardwareWarning

SUGGESTED_BACKBONES = (
    "jhu-clsp/mmBERT-base", "xlm-roberta-base", "bert-base-multilingual-cased", "deepset/gbert-base",
)


def transformers_module():
    return require("transformers", feature="supervised pair scoring", extra="finetune", min_version="4.48")


@contextlib.contextmanager
def quiet_transformers(enabled: bool = True):
    """Silence transformers' load reports and progress bars (e.g. 'newly initialized' notes)."""
    if not enabled:
        yield
        return
    transformers = transformers_module()
    logging = transformers.utils.logging
    level = logging.get_verbosity()
    bars = logging.is_progress_bar_enabled()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pin_memory.*")
            yield
    finally:
        logging.set_verbosity(level)
        if bars:
            logging.enable_progress_bar()


def dtype_kwargs(dtype) -> dict:
    """``dtype=`` (transformers >= 4.56) or ``torch_dtype=`` (older versions)."""
    if dtype is None:
        return {}
    transformers = transformers_module()
    key = "dtype" if Version(transformers.__version__.split("+")[0]) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


def check_sequence_classification(backbone: str, *, revision: str | None = None,
                                  trust_remote_code: bool = False):
    """Load only the config of ``backbone`` and check it can be a sequence classifier.

    Raises a :class:`ConfigurationError` *before* any weights are downloaded.
    """
    transformers = transformers_module()
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    )

    try:
        config = AutoConfig.from_pretrained(backbone, revision=revision, trust_remote_code=trust_remote_code)
    except OSError as exc:
        raise ModelNotAvailableError(
            f"Could not load the model {backbone!r}. Check the name (see huggingface.co) and "
            "your internet connection; offline, download the model first with "
            "simchology.download_resources() or point to a local folder."
        ) from exc
    except ValueError as exc:
        raise ConfigurationError(
            f"transformers {transformers.__version__} does not know the architecture of "
            f"{backbone!r}. Upgrade transformers, or pass method_params={{'trust_remote_code': True}} "
            f"if you trust the model's code. Details: {exc}"
        ) from exc
    supported = config.model_type in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
    auto_map = getattr(config, "auto_map", None) or {}
    if not supported and trust_remote_code and "AutoModelForSequenceClassification" in auto_map:
        supported = True
    if not supported:
        raise ConfigurationError(
            f"The model {backbone!r} (architecture {config.model_type!r}) cannot be used as a "
            f"sequence classifier with transformers {transformers.__version__}. Choose another "
            f"backbone, e.g. {', '.join(repr(b) for b in SUGGESTED_BACKBONES)}."
        )
    return config


def load_tokenizer(backbone: str, *, revision=None, trust_remote_code=False):
    transformers_module()
    from transformers import AutoTokenizer

    with quiet_transformers():
        return AutoTokenizer.from_pretrained(backbone, revision=revision, trust_remote_code=trust_remote_code)


def load_sequence_classifier(source: str, *, num_labels: int | None = None, regression: bool = False,
                             revision=None, trust_remote_code=False, dtype=None,
                             require_head: bool = False, **extra):
    """Load ``AutoModelForSequenceClassification``; optionally insist on a trained head."""
    transformers_module()
    from transformers import AutoModelForSequenceClassification

    kwargs: dict[str, Any] = dict(revision=revision, trust_remote_code=trust_remote_code, **dtype_kwargs(dtype), **extra)
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
        kwargs["problem_type"] = "regression" if regression else "single_label_classification"
        if not regression and num_labels == 2:
            kwargs["id2label"] = {0: "dissimilar", 1: "similar"}
            kwargs["label2id"] = {"dissimilar": 0, "similar": 1}
    with quiet_transformers():
        model, info = AutoModelForSequenceClassification.from_pretrained(
            source, output_loading_info=True, **kwargs
        )
    if require_head:
        missing = [k for k in info.get("missing_keys", []) if k.split(".")[0] in ("classifier", "score", "pre_classifier")]
        if missing:
            raise ConfigurationError(
                f"{source!r} has no trained classification head (missing: {', '.join(sorted(missing))}), "
                "so it cannot score pairs without training. Use method='cross_encoder' with coded "
                "data to fine-tune it, or choose a trained cross-encoder."
            )
    return model


def make_compute_metrics(kind: str):
    """Metrics computed on the development pairs after every epoch."""

    def compute(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if kind == "classification":
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            preds = logits.argmax(axis=1)
            y = labels.astype(int)
            return {
                "f1": f1_score(y, preds, zero_division=0),
                "precision": precision_score(y, preds, zero_division=0),
                "recall": recall_score(y, preds, zero_division=0),
                "accuracy": accuracy_score(y, preds),
            }
        from scipy.stats import pearsonr, spearmanr

        preds = logits.reshape(len(logits), -1)[:, 0]
        if np.std(preds) == 0 or np.std(labels) == 0:
            rho = r = -1.0
        else:
            rho = float(np.nan_to_num(spearmanr(labels, preds)[0], nan=-1.0))
            r = float(np.nan_to_num(pearsonr(labels, preds)[0], nan=-1.0))
        return {"spearman": rho, "pearson": r, "mse": float(np.mean((preds - labels) ** 2))}

    return compute


def _weighted_trainer(trainer_cls, weights: Sequence[float]):
    torch = require("torch", feature="class-weighted training")

    w = torch.tensor(list(weights), dtype=torch.float)

    class WeightedTrainer(trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.float()
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=w.to(logits.device))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def train_model(
    model,
    tokenizer,
    train_ds,
    dev_ds,
    *,
    kind: str,
    params: dict,
    device: str,
    seed: int,
    verbose: bool,
    class_weights: Sequence[float] | None = None,
    gradient_checkpointing: bool = False,
) -> tuple[Any, dict]:
    """Train with the transformers Trainer; return the (best) model and a training summary."""
    transformers_module()
    from transformers import (
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from ..hardware.runtime import training_runtime_kwargs

    tmp_dir = None
    output_dir = params.get("output_dir")
    if not output_dir:
        tmp_dir = tempfile.mkdtemp(prefix="simchology-train-")
        output_dir = tmp_dir
    has_eval = dev_ds is not None and len(dev_ds) > 0
    metric = params.get("metric_for_best", "auto")
    if metric == "auto":
        metric = "f1" if kind == "classification" else "spearman"

    runtime_kwargs = training_runtime_kwargs(device)
    bf16 = params.get("bf16", "auto")
    if bf16 is False:
        runtime_kwargs["bf16"] = False
    elif bf16 is True and not runtime_kwargs.get("bf16"):
        warn("bf16=True was requested but this device does not support bf16 training; "
                      "training in full precision instead.", SimchologyHardwareWarning)
    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=float(params["epochs"]),
        per_device_train_batch_size=int(params["batch_size"]),
        per_device_eval_batch_size=int(params["eval_batch_size"]),
        gradient_accumulation_steps=int(params["grad_accum"]),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        seed=seed,
        data_seed=seed,
        report_to="none",
        disable_tqdm=not verbose,
        logging_strategy="epoch" if verbose else "no",
        save_only_model=True,
        gradient_checkpointing=bool(gradient_checkpointing),
        # Trainer infers label names from forward's signature and drops all other input
        # columns; models that take labels only via **kwargs (e.g. Qwen3.5) would lose them.
        label_names=["labels"],
        **runtime_kwargs,
    )
    if gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if has_eval:
        kwargs.update(eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                      metric_for_best_model=metric, greater_is_better=metric != "mse",
                      save_total_limit=2)
    else:
        kwargs.update(eval_strategy="no", save_strategy="no")
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    args = TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})

    patience = int(params.get("early_stopping_patience") or 0)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if has_eval and patience else []
    trainer_cls = _weighted_trainer(Trainer, class_weights) if class_weights is not None else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if has_eval else None,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(kind) if has_eval else None,
        callbacks=callbacks,
    )
    if not verbose:
        from transformers.trainer_callback import PrinterCallback, ProgressCallback

        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
    try:
        with quiet_transformers():
            trainer.train()
        state = trainer.state
        summary = {
            "epochs_trained": float(state.epoch or 0),
            "best_metric": None if state.best_metric is None else float(state.best_metric),
            "metric": metric if has_eval else None,
            "steps": int(state.global_step),
        }
        model = trainer.model
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return model, summary


def predict_logits(model, batches, *, device: str):
    """Run ``model`` on tokenised ``batches`` (dicts of tensors) without gradients."""
    torch = require("torch", feature="supervised pair scoring")
    model.eval()
    out = []
    with torch.no_grad():
        for enc in batches:
            enc = {k: v.to(device) for k, v in enc.items()}
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


def logits_to_scores(logits: np.ndarray, activation: str) -> np.ndarray:
    """Turn logits into one score per pair.

    ``"softmax"``: probability of the last class; ``"sigmoid"``: sigmoid of the single
    logit; ``"identity"``: the first logit unchanged (regression outputs).
    """
    logits = np.asarray(logits, dtype=float)
    if activation == "softmax":
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        return (p / p.sum(axis=1, keepdims=True))[:, -1]
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-logits[:, 0]))
    return logits[:, 0]


def release_memory(device: str) -> None:
    torch = require("torch", feature="supervised pair scoring")
    import gc

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()
