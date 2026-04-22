import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()

    # YAML 설정 파일만 정확하게 전달합니다.
    sys.argv = [
        "mace_run_train",
        "--config", config_file_path
    ]
    
    print(f"Starting MACE finetuning with config: {config_file_path}")
    mace_run_train_main()

if __name__ == "__main__":
    train_mace("config.yml")
