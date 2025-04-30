# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import runpy
import sys
from pathlib import Path

import pytest
import ray
import torch

from omegaconf import OmegaConf
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    CKPT_COMPONENT_MAP,
    dummy_alpaca_dataset_config,
    MODEL_TEST_CONFIGS,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)

from torchrl.collectors.collectors import WeightUpdateSenderBase
from torchtune.dev.rl.workers import MetricLoggerWorker, TrainingWorker
from vllm.utils import get_ip, get_open_port


@ray.remote(num_cpus=4, num_gpus=1)
class DummyParamServer(WeightUpdateSenderBase):
    def __init__(self, env_vars):
        super().__init__()
        torch.cuda.set_device(torch.device("cuda", 0))

        for k, v in env_vars.items():
            os.environ[k] = str(v)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

    def register_model_metadata(self, model_metadata):
        self.state_dicßt = dict()
        for k, (dtype, shape) in model_metadata.items():
            self.state_dict[k] = torch.empty(shape, dtype=dtype, device="cuda")
        self.version = 0
        self.version_tensor = torch.tensor([self.version], device="cuda")

    def receive_from_trainer(self):
        for v in self.state_dict.values():
            torch.distributed.recv(v, src=0)

    def acquire_state_dict_lock(self):
        pass

    def release_state_dict_lock(self):
        pass

    def _get_server_weights(self):
        pass

    def _maybe_map_weights(self, server_weights):
        pass

    def _sync_weights_with_worker(self, worker_id, server_weights):
        pass

    def all_worker_ids(self):
        pass


@pytest.fixture(params=[2, 1], autouse=True)
def world_size(request):
    size = request.param + 1  # add 1 for param server
    num_cpus = 8 * request.param + 4 + 1  # trainers + param + metrics
    ray.init(num_cpus=num_cpus, num_gpus=size)
    yield size
    ray.shutdown()


class TestAsyncGRPOTrainerWorker:
    def _get_test_config_overrides(self, epochs: int = 2):
        return [
            "dataset.train_on_input=False",
            "orchestration.num_steps=4",
            "seed=9",
            "log_every_n_steps=1",
            "training.enable_activation_checkpointing=False",
            "training.enable_activation_offloading=False",
            "training.dtype=fp32",
            "training.optimizer=torch.optim.AdamW",
            "training.optimizer.lr=2e-5",
            "training.steps_before_sync=2",
            "training.device_type=cuda",
            f"training.epochs={epochs}",
            "training.ppo_epochs=1",
            "training.clip_grad_norm=1.0",
            "training.ppo_epochs=1",
            "training.resume_from_checkpoint=False",
            "training.save_every_n_epochs=1",
        ]

    def _fetch_expected_loss_values_multi_rank(self, model_type):
        loss_values_map = {
            "llama2": [10.5209, 10.5217, 10.4945, 10.5136],
            "llama3": [11.9839, 11.9684, 11.9596, 11.93656],
        }
        return loss_values_map[model_type]

    def _fetch_expected_loss_values_single_rank(self, model_type):
        loss_values_map = {
            "llama2": [10.5051, 10.5572, 10.4780, 10.5678],
            "llama3": [11.9742, 12.0049, 11.9382, 12.0464],
        }
        return loss_values_map[model_type]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps, cpu_offload",
        [
            ("llama3/8B_full", "llama3", "tune", 1, 4, False),
            ("llama3/8B_full", "llama3", "tune", 4, 1, True),
        ],
    )
    @gpu_test(gpu_count=3)
    def test_loss(
        self,
        micro_batch_size,
        gradient_accumulation_steps,
        config,
        model_type,
        ckpt_type,
        cpu_offload,
        world_size,
        tmpdir,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        overrides = f"""
            training.batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            "fsdp_cpu_offload={cpu_offload}" \
        """.split()
        model_config = MODEL_TEST_CONFIGS[model_type]
        overrides += self._get_test_config_overrides() + model_config

        cfg_path = Path(__file__).parents[2] / "recipes/configs" / config
        base_cfg = OmegaConf.load(str(cfg_path) + ".yaml")
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)

        # Create Trainers
        trainers = []
        replay_buffer = [torch.randn((micro_batch_size, 100, 16)) for _ in range(10)]
        ip, port = get_ip(), get_open_port()
        num_trainers = world_size - 1
        for i in range(num_trainers):
            env_vars = {
                "RANK": str(i),
                "WORLD_SIZE": world_size,
                "MASTER_ADDR": ip,
                "MASTER_PORT": port,
            }
            trainer = TrainingWorker.remote(
                cfg,
                env_vars,
                replay_buffer[i::num_trainers],
            )
            trainers.append(trainer)

        # Register dummy parameter server
        env_vars["RANK"] = str(num_trainers)
        param_server = DummyParamServer.remote(env_vars)
        trainers[0].register_parameter_server.remote(param_server)

        # Add Metric Logging
        logger = MetricLoggerWorker.remote(cfg)
        logger_handles = []
        for worker in trainers:
            logger_handles.append(worker.set_metric_logger.remote(logger))
        ray.get(logger_handles)  # wait for all calls to complete

        model_metadata = ray.get(trainers[0].get_model_metadata.remote())
        param_server.register_model_metadata.remote(model_metadata)

        # Train
        trainer_handles = [worker.train.remote() for worker in trainers]
        ray.get(trainer_handles)
        ray.get(trainers[0].cleanup.remote())

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values_multi_rank(model_type)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )
