# Здесь будет прописана основная работа программы


# сюда можно докинуть hydra и юзать как параметр название архитектуры

import hydra
from omegaconf import DictConfig
from src.arch.cgan.runner import runner_cgan


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    if cfg.model.arch == "CGAN":
        runner_cgan(cfg)


if __name__ == "__main__":
    main()
