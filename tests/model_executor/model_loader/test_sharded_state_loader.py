# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fnmatch
import multiprocessing as mp
import os
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import ShardedStateLoader
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    ignore_eos=True,
)


def test_filter_subtensors():
    state_dict = {
        "a": torch.empty(2),
        "b": torch.empty((2, 4)),
        "c": torch.empty((2, 4, 8)),
    }
    state_dict.update(
        {
            "x": state_dict["b"],
            "y": state_dict["c"][1, 2, :],
            "z": state_dict["c"][1, :, 4],
        }
    )
    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
    for key, tensor in filtered_state_dict.items():
        # NOTE: don't use `equal` here, as the tensor might contain NaNs
        assert tensor is state_dict[key]


def _touch_shards(directory, pattern, tp_size, parts=2):
    for rank in range(tp_size):
        for part in range(parts):
            name = pattern.format(rank=rank, part=part)
            open(os.path.join(directory, name), "w").close()


@pytest.mark.parametrize("pattern", [None, "custom-{rank}-{part}.safetensors"])
def test_saved_tp_size_counts_ranks(pattern):
    extra_config = {"pattern": pattern} if pattern else None
    loader = ShardedStateLoader(
        LoadConfig(load_format="sharded_state", model_loader_extra_config=extra_config)
    )
    with TemporaryDirectory() as directory:
        _touch_shards(directory, loader.pattern, tp_size=4)
        open(os.path.join(directory, "config.json"), "w").close()
        assert loader._saved_tp_size(directory) == 4


def test_saved_tp_size_returns_none_when_undeterminable():
    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    with TemporaryDirectory() as directory:
        assert loader._saved_tp_size(directory) is None

    rankless = ShardedStateLoader(
        LoadConfig(
            load_format="sharded_state",
            model_loader_extra_config={"pattern": "model-part-{part}.safetensors"},
        )
    )
    with TemporaryDirectory() as directory:
        open(os.path.join(directory, "model-part-0.safetensors"), "w").close()
        assert rankless._saved_tp_size(directory) is None


def _model_config_stub(directory):
    """Minimal stand-in for ModelConfig.

    load_weights() reads exactly two attributes before the guard fires, and
    building a real ModelConfig would require a downloadable model for a check
    that never gets as far as looking at one.
    """
    return SimpleNamespace(model=directory, model_weights=None)


@pytest.mark.parametrize("saved_tp_size, running_tp_size", [(4, 2), (2, 4)])
def test_load_weights_rejects_tp_mismatch(monkeypatch, saved_tp_size, running_tp_size):
    """The shrinking direction is the one that used to corrupt silently.

    Saved at TP=4 and served at TP=2, every rank finds files that are half the
    size of its parameters, load_weights() narrows each one, the missing-keys
    check at the end passes, and the model serves with half of every tensor
    uninitialized. The only signal was a per-tensor logger.warning.
    """
    import vllm.distributed as dist

    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    with TemporaryDirectory() as directory:
        _touch_shards(directory, loader.pattern, tp_size=saved_tp_size)
        monkeypatch.setattr(
            dist, "get_tensor_model_parallel_world_size", lambda: running_tp_size
        )
        with pytest.raises(ValueError) as excinfo:
            loader.load_weights(torch.nn.Module(), _model_config_stub(directory))

    message = str(excinfo.value)
    assert f"tensor_parallel_size={saved_tp_size}" in message
    assert f"tensor_parallel_size={running_tp_size}" in message


def test_load_weights_allows_matching_tp(monkeypatch):
    """The guard must not fire when the sizes agree.

    The load still fails afterwards -- these are empty placeholder files, not
    real safetensors -- so this asserts on which failure is reached, which is
    what distinguishes "guard passed" from "guard fired".
    """
    import vllm.distributed as dist

    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    with TemporaryDirectory() as directory:
        _touch_shards(directory, loader.pattern, tp_size=2)
        monkeypatch.setattr(dist, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_tensor_model_parallel_rank", lambda: 0)
        with pytest.raises(Exception) as excinfo:
            loader.load_weights(torch.nn.Module(), _model_config_stub(directory))

    assert "Sharded checkpoint" not in str(excinfo.value), (
        "TP guard fired on a checkpoint whose size matches the running instance"
    )


@pytest.fixture(scope="module")
def llama_3p2_1b_files():
    input_dir = hf_api().snapshot_download(
        "meta-llama/Llama-3.2-1B-Instruct", ignore_patterns=["*.bin*", "original/*"]
    )

    yield input_dir


def _run_writer(input_dir, output_dir, weights_patterns, **kwargs):
    llm_sharded_writer = LLM(model=input_dir, **kwargs)

    # Dump worker states to output directory
    llm_sharded_writer.llm_engine.engine_core.save_sharded_state(path=output_dir)

    # Copy metadata files to output directory
    for file in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, file)):
            shutil.copytree(
                os.path.join(input_dir, file), os.path.join(output_dir, file)
            )
        elif not any(fnmatch.fnmatch(file, ext) for ext in weights_patterns):
            shutil.copy(os.path.join(input_dir, file), output_dir)


def _run_generate(input_dir, queue: mp.Queue, **kwargs):
    llm = LLM(model=input_dir, **kwargs)
    gen = llm.generate(prompts, sampling_params)
    queue.put([g.outputs[0].__dict__ for g in gen])
    queue.close()
    queue.join_thread()


@pytest.mark.parametrize("enable_lora", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_sharded_state_loader(
    enable_lora, tp_size, num_gpus_available, llama_3p2_1b_files
):
    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    weights_patterns = ("*.safetensors",)
    gpu_memory_utilization = 0.8
    input_dir = llama_3p2_1b_files
    ctx = mp.get_context("spawn")

    platform_args = {}
    if current_platform.is_rocm() or current_platform.is_xpu():
        platform_args["max_num_seqs"] = 1

    # Run in separate processes for memory & CUDA isolation
    with TemporaryDirectory() as output_dir:
        p = ctx.Process(
            target=_run_writer,
            args=(input_dir, output_dir, weights_patterns),
            kwargs=dict(
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                **platform_args,
            ),
        )
        p.start()
        p.join()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(input_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_before = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        queue = ctx.Queue()

        p = ctx.Process(
            target=_run_generate,
            args=(output_dir, queue),
            kwargs=dict(
                enable_lora=enable_lora,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tp_size,
                load_format="sharded_state",
                **platform_args,
            ),
        )
        p.start()
        # Call queue.get() before p.join() to prevent deadlock:
        # If p.join() is called before queue.get() and the queue is full,
        # the child process may block while writing to the queue and never
        # terminate, causing the parent to wait indefinitely on p.join().
        # See: https://github.com/vllm-project/vllm/pull/22371#discussion_r2257773814
        out_after = queue.get()
        p.join()
        queue.close()
        queue.join_thread()

        assert out_before == out_after
