import warnings
warnings.filterwarnings("ignore")
from mace.cli.fine_tuning_select import main as mace_ft_select_main
import sys
import logging

def run_fine_tuning_select():
    logging.getLogger().handlers.clear()
    
    sys.argv = [
        "fine_tuning_select",  # 실행 프로그램 이름 (더미 값)
        "--configs_pt", "/home/kyunghun/02_ft_test/OLD/replay-data-mh-1-omat-pbe.xyz",
        "--configs_ft", "train.extxyz",
        "--num_samples", "500",
        "--subselect", "fps",
        "--filtering_type", "combinations",
        "--output", "LAOC_selected_replay.xyz",
        "--device", "cuda",
        "--default_dtype", "float64",
    ]
    
    # main 함수 실행
    mace_ft_select_main()
    
# 함수 호출
run_fine_tuning_select()
