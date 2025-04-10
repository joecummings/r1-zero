import ray
from torchtune import config


@ray.remote(num_cpus=1, num_gpus=0)
class MetricLoggerActor:
    def __init__(self, cfg):
        self.logger = config.instantiate(cfg.metric_logger)
        self.logger.log_config(cfg)

    def log_dict(self, log_dict, step=None):
        # allowing actors to use their own step counters
        self.logger.log_dict(log_dict, step=step)

    def close(self):
        if hasattr(self.logger, "close"):
            self.logger.close()
