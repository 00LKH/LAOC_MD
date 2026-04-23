## MD ## MD ## MD ## MD ## MD ## MD ## MD ## MD ## MD ## MD ## MD ## MD 
import os
import glob
import datetime
import numpy as np
import torch
import random

from ase import units
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator
from ase.data import covalent_radii, atomic_numbers
from ase.constraints import Hookean

# ==========================================
# 1. 시뮬레이션 설정 (Configuration)
# ==========================================
seed_value = 2026
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# 모델 및 경로 설정
model_path = "/home/kyunghun/00_cu_interface/99_archieve/02_mace/mace-mh-1.model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# MD 파라미터
temperature_K = 1200
timestep_fs = 2.0
nvt_steps = 1001
log_interval = 10
nvt_tdamp_fs = 100

# Hookean 제약 조건 파라미터
hookean_tolerance = 0.5  
hookean_k = 10.0         
rt_offset = 0.2          

if not hasattr(units, 'ps'):
    units.ps = 1000 * units.fs

base_dir = os.path.abspath(".")

# ==========================================
# 2. Helper 함수 정의
# ==========================================

# [개선 2] 현재 궤적(atoms_to_constrain)과 초기 기준 구조(atoms_for_reference)를 분리하여 받음
def apply_hookean_constraints(atoms_to_constrain, atoms_for_reference, k_val, tolerance, rt_extra):
    r_al = covalent_radii[atomic_numbers['Al']]
    r_o  = covalent_radii[atomic_numbers['O']]
    bond_cutoff = r_al + r_o + tolerance
    rt_val = bond_cutoff + rt_extra

    try:
        chain_ids = atoms_for_reference.get_array('chain_id')
    except KeyError:
        print("경고: 'chain_id' 배열이 존재하지 않습니다. 폴백 모드로 동작합니다.")
        chain_ids = [0] * len(atoms_for_reference)

    al_indices = [atom.index for atom in atoms_for_reference if atom.symbol == 'Al']
    o_indices = [atom.index for atom in atoms_for_reference if atom.symbol == 'O']

    constraints = []
    hookean_pairs = [] 
    
    for i in al_indices:
        for j in o_indices:
            if chain_ids[i] > 0 and chain_ids[i] == chain_ids[j]:
                # 페어(결합) 여부는 변형되지 않은 '원본 기준 구조'의 거리를 바탕으로 판단
                dist = atoms_for_reference.get_distance(i, j, mic=True)
                
                if dist <= bond_cutoff:
                    constraints.append(Hookean(a1=i, a2=j, k=k_val, rt=rt_val))
                    hookean_pairs.append((i, j))
    
    # 묶어줄 페어가 결정되면, 실제 MD를 수행할 구조(현재 프레임)에 제약 조건 적용
    atoms_to_constrain.set_constraint(constraints)
    return hookean_pairs

def prepare_simulation(stage_name, target_steps, initial_atoms_path, input_stem):
    extxyz_filename = f"{input_stem}_{stage_name}.extxyz"
    
    if os.path.exists(extxyz_filename):
        try:
            history = read(extxyz_filename, index=":")
            steps_already_done = max(0, (len(history) - 1) * log_interval)
            if steps_already_done >= target_steps:
                return history[-1], 0, steps_already_done, 'a'
            return history[-1], target_steps - steps_already_done, steps_already_done, 'a'
        except:
            pass
    
    atoms = read(initial_atoms_path)
    return atoms, target_steps, 0, 'w'

def attach_loggers(dyn, atoms, log_filename, extxyz_filename, file_mode, hookean_pairs, start_step=0, interval=100):
    def write_extxyz():
        atoms.write(extxyz_filename, format="extxyz", append=True)

    dyn.attach(write_extxyz, interval=interval)
    
    logfile = open(log_filename, file_mode)
    if file_mode == 'w':
        header = f"{'Step':>8}  {'Time(ps)':>12}  {'Temp(K)':>12}  {'Max_Al_O(A)':>15}  {'Timestamp':<19}\n"
        logfile.write(header)
    
    def log_status():
        internal_step = dyn.get_number_of_steps()
        if file_mode == 'a' and internal_step == 0: return
        current_step = internal_step + start_step
        current_time = (dyn.get_time() + (start_step * dyn.dt)) / units.ps
        temperature = atoms.get_temperature()

        max_dist = 0.0
        if hookean_pairs:
            distances = [atoms.get_distance(i, j, mic=True) for i, j in hookean_pairs]
            max_dist = max(distances)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{current_step:>8d}  {current_time:>12.4f}  {temperature:>12.2f}  {max_dist:>15.4f}  {timestamp}\n"
        logfile.write(log_line)
        logfile.flush()

    dyn.attach(log_status, interval=interval)
    return logfile

# ==========================================
# 3. 메인 실행 (Main Execution)
# ==========================================
print("Loading MACE Calculator...")
calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64", head="omat_pbe")

all_folders = sorted([
    f for f in os.listdir(base_dir) 
    if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("structure_")
])

mid_index = len(all_folders) // 2

# # 상단
# target_folders = all_folders[:mid_index] 
# # 하단
# target_folders = all_folders[mid_index:] 

# All
target_folders = all_folders[:]

print(f"Targeting {len(target_folders)} folders: {target_folders}")

for folder_idx, folder_name in enumerate(target_folders):
    work_dir = os.path.join(base_dir, folder_name)
    if not os.path.isdir(work_dir): continue
        
    print(f"\nEntering Directory: {work_dir}")
    try:
        os.chdir(work_dir)
        
        # [개선 1] _nvt 가 포함된 궤적 파일은 제외하고 순수 초기 구조 파일만 탐색
        input_files = [f for f in glob.glob("structure_*.extxyz") if "_nvt" not in f]
        if not input_files:
            print("  -> 적절한 초기 구조 파일을 찾을 수 없습니다.")
            continue
            
        input_file_path = input_files[0]
        input_stem = os.path.splitext(input_file_path)[0]
        
        # 항상 고정된 원본 초기 구조를 별도로 읽어둠 (Hookean 페어 판별용)
        initial_atoms = read(input_file_path)
        
        atoms_nvt, steps_to_run_nvt, steps_done_nvt, mode_nvt = prepare_simulation(
            "nvt", nvt_steps, input_file_path, input_stem
        )
        atoms_nvt.calc = calc
        
        # [개선 2 적용] atoms_nvt에 제약을 걸되, 페어(짝) 판별은 initial_atoms 기준으로 수행
        hookean_pairs = apply_hookean_constraints(atoms_nvt, initial_atoms, hookean_k, hookean_tolerance, rt_offset)
        print(f"  -> Applied {len(hookean_pairs)} Hookean constraints (Al-O).")
        
        if steps_done_nvt == 0:
            # [개선 3] 각 폴더마다 고유한 시드 적용하여 통계적 앙상블 다양성 확보
            unique_seed = seed_value + folder_idx
            rng = np.random.default_rng(unique_seed)
            MaxwellBoltzmannDistribution(atoms_nvt, temperature_K=temperature_K, rng=rng)
            Stationary(atoms_nvt)
            
        if steps_to_run_nvt > 0:
            dyn_nvt = NoseHooverChainNVT(
                atoms_nvt,
                timestep=timestep_fs * units.fs,
                temperature_K=temperature_K,
                tdamp=nvt_tdamp_fs * units.fs
            )
            extxyz_out = f"{input_stem}_nvt.extxyz"
            log_nvt = attach_loggers(
                dyn_nvt, atoms_nvt, 
                f"md_{input_stem}_nvt.log", 
                extxyz_out, 
                mode_nvt, 
                hookean_pairs, 
                start_step=steps_done_nvt, 
                interval=log_interval
            )
            dyn_nvt.run(steps=steps_to_run_nvt)
            log_nvt.close()

    except Exception as e:
        print(f"  -> Error: {e}")
    finally:
        os.chdir(base_dir)

print("\nAll tasks finished.")
