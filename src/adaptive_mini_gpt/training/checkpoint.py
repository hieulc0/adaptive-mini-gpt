"""Checkpointing mechanisms for Adaptive Mini-GPT with atomic publishing, backup, and recovery."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages atomic saving, backup synchronization, manifest indexing, and recovery."""

    def __init__(
        self,
        save_dir: str | Path,
        backup_dir: str | Path | None = None,
        keep_last_n: int = 5,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.manifest_path = self.save_dir / "manifest.json"

    def _get_rng_states(self) -> dict[str, Any]:
        """Capture all relevant RNG states for deterministic resumption."""
        states: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        return states

    def _restore_rng_states(self, states: dict[str, Any]) -> None:
        """Restore RNG states."""
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch_cpu" in states and isinstance(states["torch_cpu"], torch.Tensor):
            torch.set_rng_state(states["torch_cpu"])
        if (
            "torch_cuda" in states
            and states["torch_cuda"] is not None
            and torch.cuda.is_available()
        ):
            # A checkpoint can move between a multi-GPU runtime and a single GPU.
            # Restore matching device indices; new devices retain their seeded state.
            cuda_states = states["torch_cuda"]
            device_count = torch.cuda.device_count()
            if len(cuda_states) != device_count:
                logger.warning(
                    "CUDA device count changed (%s -> %s); restoring matching RNG states only",
                    len(cuda_states),
                    device_count,
                )
            torch.cuda.set_rng_state_all(cuda_states[:device_count])

    def _smoke_test_checkpoint(self, path: Path) -> None:
        """Smoke-load the file to guarantee it is valid and non-corrupted."""
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint file {path} is empty or missing.")
        try:
            _ = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Checkpoint verification failed on {path}: {e}") from e

    def _copy_verified_to_backup(self, src: Path, dest_dir: Path) -> Path:
        """Copy a file to backup directory using an atomic temporary write and verification."""
        dest_target = dest_dir / src.name
        dest_temp = dest_dir / f"{src.name}.tmp"
        shutil.copy2(src, dest_temp)
        self._smoke_test_checkpoint(dest_temp)
        os.replace(dest_temp, dest_target)
        return dest_target

    def save_checkpoint(
        self,
        checkpoint_name: str,
        model: nn.Module,
        optimizer: Optimizer,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer: Any | None = None,
        dataset_fingerprint: str = "",
        step: int = 0,
        tokens_seen: int = 0,
        data_position: int = 0,
        best_val_loss: float = float("inf"),
        run_id: str = "run",
        scaler: Any | None = None,
        scheduler: LRScheduler | None = None,
        is_best: bool = False,
    ) -> Path:
        """Save a complete checkpoint atomically and update manifest and optional backup."""
        save_start = time.time()
        checkpoint_data: dict[str, Any] = {
            "format_version": "1.0",
            "run_id": run_id,
            "step": step,
            "tokens_seen": tokens_seen,
            "data_position": data_position,
            "seed": training_config.seed,
            "best_val_loss": best_val_loss,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "tokenizer": (
                tokenizer.to_dict()
                if tokenizer is not None and hasattr(tokenizer, "to_dict")
                else None
            ),
            "tokenizer_fingerprint": (
                tokenizer.get_fingerprint()
                if tokenizer is not None and hasattr(tokenizer, "get_fingerprint")
                else ""
            ),
            "dataset_fingerprint": dataset_fingerprint,
            "rng_states": self._get_rng_states(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        }

        # 1. Save versioned checkpoint: step_XXXXXX.pt
        versioned_name = f"step_{step:06d}.pt"
        versioned_path = self.save_dir / versioned_name
        versioned_temp = self.save_dir / f"{versioned_name}.tmp"

        torch.save(checkpoint_data, versioned_temp)
        self._smoke_test_checkpoint(versioned_temp)
        os.replace(versioned_temp, versioned_path)

        # 2. Save named checkpoint (e.g. latest.pt)
        target_path = self.save_dir / checkpoint_name
        target_temp = self.save_dir / f"{checkpoint_name}.tmp"
        torch.save(checkpoint_data, target_temp)
        self._smoke_test_checkpoint(target_temp)
        os.replace(target_temp, target_path)

        # 3. If best, save best.pt
        if is_best:
            best_path = self.save_dir / "best.pt"
            best_temp = self.save_dir / "best.pt.tmp"
            torch.save(checkpoint_data, best_temp)
            self._smoke_test_checkpoint(best_temp)
            os.replace(best_temp, best_path)

        # 4. Prune old versioned checkpoints locally
        self._prune_old_checkpoints(self.save_dir)

        # 5. Update local manifest atomically
        self._update_manifest(
            manifest_dir=self.save_dir,
            latest_path=target_path,
            versioned_path=versioned_path,
            step=step,
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            run_id=run_id,
        )

        # 6. Publish to backup directory (e.g. Google Drive) if configured
        if self.backup_dir is not None:
            try:
                self._copy_verified_to_backup(versioned_path, self.backup_dir)
                self._copy_verified_to_backup(target_path, self.backup_dir)
                if is_best and (self.save_dir / "best.pt").exists():
                    self._copy_verified_to_backup(self.save_dir / "best.pt", self.backup_dir)
                self._prune_old_checkpoints(self.backup_dir)
                self._update_manifest(
                    manifest_dir=self.backup_dir,
                    latest_path=self.backup_dir / checkpoint_name,
                    versioned_path=self.backup_dir / versioned_name,
                    step=step,
                    tokens_seen=tokens_seen,
                    best_val_loss=best_val_loss,
                    run_id=run_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to publish checkpoint to backup dir %s: %s", self.backup_dir, e
                )

        duration = time.time() - save_start
        logger.info("Saved checkpoint %s in %.3f seconds.", checkpoint_name, duration)
        return target_path

    def _prune_old_checkpoints(self, directory: Path) -> None:
        """Keep only the latest N versioned checkpoints in a directory."""
        if self.keep_last_n <= 0:
            return
        versioned_files = sorted(
            directory.glob("step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(versioned_files) > self.keep_last_n:
            to_remove = versioned_files[: -self.keep_last_n]
            for f in to_remove:
                try:
                    f.unlink()
                except OSError:
                    pass

    def _update_manifest(
        self,
        manifest_dir: Path,
        latest_path: Path,
        versioned_path: Path,
        step: int,
        tokens_seen: int,
        best_val_loss: float,
        run_id: str,
    ) -> None:
        manifest_file = manifest_dir / "manifest.json"
        history: list[dict[str, Any]] = []

        if manifest_file.exists():
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    old_data = json.load(f)
                    history = old_data.get("history", [])
            except Exception:
                history = []

        history.append(
            {
                "step": step,
                "filename": versioned_path.name,
                "best_val_loss": best_val_loss,
            }
        )
        # Keep history capped at 50 records
        history = history[-50:]

        best_path = manifest_dir / "best.pt"
        manifest_data = {
            "run_id": run_id,
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "latest_checkpoint": str(latest_path.name),
            "best_checkpoint": str(best_path.name) if best_path.exists() else None,
            "history": history,
        }
        temp_manifest = manifest_dir / "manifest.json.tmp"
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(temp_manifest, manifest_file)

    @classmethod
    def load_tokenizer(
        cls,
        raw_checkpoint: dict[str, Any],
        model_config: Any | None = None,
    ) -> Any:
        """Instantiate and validate tokenizer from checkpoint, supporting Character and BPE."""
        tok_dict = raw_checkpoint.get("tokenizer")
        if tok_dict and (tok_dict.get("tokenizer_type") == "bpe" or "merges" in tok_dict):
            from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer

            tokenizer = BPETokenizer.from_dict(tok_dict)
        elif tok_dict:
            from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer

            tokenizer = CharacterTokenizer.from_dict(tok_dict)
        elif raw_checkpoint.get("tokenizer_vocab"):
            from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer

            tokenizer = CharacterTokenizer.from_dict({"vocab": raw_checkpoint["tokenizer_vocab"]})
        else:
            from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer

            tokenizer = CharacterTokenizer(use_base_ascii=True).fit("")

        expected_fp = raw_checkpoint.get("tokenizer_fingerprint")
        if expected_fp and hasattr(tokenizer, "get_fingerprint"):
            actual_fp = tokenizer.get_fingerprint()
            if actual_fp != expected_fp:
                raise ValueError(
                    f"Tokenizer fingerprint mismatch! Checkpoint recorded '{expected_fp}', "
                    f"but rehydrated tokenizer produced '{actual_fp}'."
                )

        if model_config is not None:
            v_size = getattr(model_config, "vocab_size", None)
            if v_size is not None and tokenizer.vocab_size != v_size:
                raise ValueError(
                    f"Lineage Mismatch: Checkpoint vocabulary size ({v_size}) "
                    f"does not match tokenizer vocabulary size ({tokenizer.vocab_size})."
                )

        return tokenizer

    @classmethod
    def load_raw_checkpoint(cls, checkpoint_path: str | Path) -> dict[str, Any]:
        """Safely load raw checkpoint dictionary from disk."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read checkpoint {path}: {e}") from e

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint at {path} does not contain a valid dictionary.")
        return checkpoint

    @classmethod
    def find_latest_valid_checkpoint(
        cls,
        path_or_dir: str | Path,
        search_backup: Path | None = None,
    ) -> Path:
        """Find the latest verified valid checkpoint, recovering if the target is corrupted.

        Args:
            path_or_dir: Path to a checkpoint file or directory containing checkpoints.
            search_backup: Optional backup directory to check if local checkpoints fail.

        Returns:
            Path to the latest valid checkpoint file.
        """
        target = Path(path_or_dir)

        # 1. If path is a specific file, test if it is valid
        if target.is_file():
            try:
                _ = cls.load_raw_checkpoint(target)
                return target
            except Exception as e:
                logger.warning(
                    "Requested checkpoint %s is corrupt (%s). Searching for valid fallback.",
                    target,
                    e,
                )
                parent_dir = target.parent
        elif target.is_dir():
            parent_dir = target
        else:
            raise FileNotFoundError(f"Path does not exist: {target}")

        # 2. Gather candidates from directory
        candidates: list[Path] = []

        # Check active pointer first (single atomic pointer)
        pointer_path = parent_dir / "active_pointer.json"
        if pointer_path.is_file():
            try:
                with open(pointer_path, encoding="utf-8") as f:
                    ptr_data = json.load(f)
                ckpt_p = ptr_data.get("checkpoint_path")
                if ckpt_p and Path(ckpt_p).is_file():
                    candidates.append(Path(ckpt_p))
            except Exception:
                pass

        # Check manifest history first
        manifest_path = parent_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest_data = json.load(f)
                latest_name = manifest_data.get("latest_checkpoint")
                if latest_name and (parent_dir / latest_name).is_file():
                    candidates.append(parent_dir / latest_name)
                for item in reversed(manifest_data.get("history", [])):
                    fname = item.get("filename")
                    if fname and (parent_dir / fname).is_file():
                        p = parent_dir / fname
                        if p not in candidates:
                            candidates.append(p)
            except Exception:
                pass

        # Also add all .pt files sorted by modification time descending
        all_pt = sorted(parent_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in all_pt:
            if p not in candidates:
                candidates.append(p)

        # 3. Test each candidate in order
        for cand in candidates:
            try:
                _ = cls.load_raw_checkpoint(cand)
                logger.info("Recovered and validated checkpoint: %s", cand)
                return cand
            except Exception:
                continue

        # 4. Check backup directory if provided and local search failed
        if search_backup is not None and search_backup.is_dir():
            logger.info("Local candidates corrupt; searching backup directory %s", search_backup)
            return cls.find_latest_valid_checkpoint(search_backup)

        raise RuntimeError(f"No valid checkpoint found in {target}")

    def restore_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        tokenizer: Any | None = None,
        dataset_fingerprint: str | None = None,
        scaler: Any | None = None,
        scheduler: LRScheduler | None = None,
        strict_dataset_match: bool = True,
        auto_recover: bool = True,
        allow_growth: bool = False,
    ) -> dict[str, Any]:
        """Restore training state and validate configuration compatibility with auto-recovery."""
        target_path = Path(checkpoint_path)
        if auto_recover:
            actual_path = self.find_latest_valid_checkpoint(
                target_path, search_backup=self.backup_dir
            )
        else:
            actual_path = target_path

        checkpoint = self.load_raw_checkpoint(actual_path)

        # 1. Validate model architecture compatibility
        ckpt_model_config = checkpoint.get("model_config", {})
        config_obj = getattr(model, "config", None)
        if config_obj is not None and hasattr(config_obj, "to_dict"):
            current_config = config_obj.to_dict()
            if allow_growth:
                critical_keys = ["d_model", "n_heads", "d_ff", "vocab_size"]
                for k in critical_keys:
                    if k in ckpt_model_config and ckpt_model_config[k] != current_config.get(k):
                        msg = (
                            f"Architecture mismatch for {k}: "
                            f"checkpoint has {ckpt_model_config[k]}, "
                            f"current model has {current_config.get(k)}"
                        )
                        raise ValueError(msg)
                curr_layers = current_config.get("n_layers", 0)
                ckpt_layers = ckpt_model_config.get("n_layers", 0)
                if "n_layers" in ckpt_model_config and curr_layers < ckpt_layers:
                    msg = (
                        f"Architecture mismatch for n_layers: current model has fewer layers "
                        f"({curr_layers}) than checkpoint ({ckpt_layers})"
                    )
                    raise ValueError(msg)
            else:
                critical_keys = ["d_model", "n_layers", "n_heads", "d_ff", "vocab_size"]
                for k in critical_keys:
                    if k in ckpt_model_config and ckpt_model_config[k] != current_config.get(k):
                        msg = (
                            f"Architecture mismatch for {k}: "
                            f"checkpoint has {ckpt_model_config[k]}, "
                            f"current model has {current_config.get(k)}"
                        )
                        raise ValueError(msg)

        # 2. Validate tokenizer compatibility
        ckpt_tok_fp = checkpoint.get("tokenizer_fingerprint", "")
        if tokenizer is not None and hasattr(tokenizer, "get_fingerprint"):
            curr_tok_fp = tokenizer.get_fingerprint()
            if ckpt_tok_fp and curr_tok_fp and ckpt_tok_fp != curr_tok_fp:
                raise ValueError(
                    f"Tokenizer fingerprint mismatch: checkpoint has {ckpt_tok_fp}, "
                    f"current tokenizer has {curr_tok_fp}"
                )

        # 3. Validate dataset compatibility
        ckpt_ds_fp = checkpoint.get("dataset_fingerprint", "")
        if dataset_fingerprint is not None and ckpt_ds_fp:
            if ckpt_ds_fp != dataset_fingerprint:
                msg = (
                    f"Dataset fingerprint mismatch: checkpoint has {ckpt_ds_fp}, "
                    f"current data has {dataset_fingerprint}"
                )
                if strict_dataset_match and not allow_growth:
                    raise ValueError(msg)

        # 4. Restore model weights
        model.load_state_dict(checkpoint["model_state_dict"], strict=not allow_growth)

        # 5. Restore optimizer state
        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            if allow_growth:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e:
                    logger.warning("Could not restore optimizer state in allow_growth mode: %s", e)
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 6. Restore scaler state
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # 7. Restore scheduler state
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            if allow_growth:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    logger.warning("Could not restore scheduler state in allow_growth mode: %s", e)
            else:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 8. Restore RNG states
        if "rng_states" in checkpoint:
            self._restore_rng_states(checkpoint["rng_states"])

        return {
            "step": checkpoint.get("step", 0),
            "tokens_seen": checkpoint.get("tokens_seen", 0),
            "data_position": checkpoint.get("data_position", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "run_id": checkpoint.get("run_id", "resumed"),
            "model_config": ckpt_model_config,
            "training_config": checkpoint.get("training_config", {}),
            "tokenizer": checkpoint.get("tokenizer", {}),
            "dataset_fingerprint": ckpt_ds_fp,
            "recovered_from": str(actual_path),
        }
