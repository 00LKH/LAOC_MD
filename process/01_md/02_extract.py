import os
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii, atomic_numbers

# ==========================================
# 1. 설정 및 파라미터
# ==========================================
# Al-Cl 결합 판단 기준 (MD 코드와 동일하게 설정)
r_al = covalent_radii[atomic_numbers['Al']]
r_cl = covalent_radii[atomic_numbers['Cl']]
AL_CL_CUTOFF = r_al + r_cl + 0.3 

BASE_DIR = "./mlmd_extxyz"
MAIN_OUTPUT_DIR = "../02_mlmd_extract/finetuning_dataset" # 병합된 최상위 저장 폴더

# 추출할 구조 개수 설정
MAX_BRIDGE_SAMPLES = 10
MAX_LOW_ENERGY_SAMPLES = 10
MAX_HIGH_ENERGY_SAMPLES = 10

import glob
# 분석할 궤적 파일 재귀적 탐색
traj_files = sorted(glob.glob(os.path.join(BASE_DIR, "**/*_nvt.extxyz"), recursive=True))

def get_bridging_cl_count(atoms):
    """현재 프레임에서 Intra-chain Bridging Cl의 개수를 반환"""
    # 체인 정보가 없는 경우를 대비한 예외 처리
    if 'chain_id' not in atoms.arrays:
        return 0
        
    chain_ids = atoms.get_array('chain_id')
    cl_i, al_j = neighbor_list('ij', atoms, {('Cl', 'Al'): AL_CL_CUTOFF})
    
    cl_to_al = {}
    for c, a in zip(cl_i, al_j):
        if atoms[c].symbol == 'Cl':
            if c not in cl_to_al: cl_to_al[c] = []
            cl_to_al[c].append(a)
    
    count = 0
    for cl_idx, al_indices in cl_to_al.items():
        if len(al_indices) == 2:
            c1, c2 = chain_ids[al_indices[0]], chain_ids[al_indices[1]]
            if c1 == c2 and c1 > 0:
                count += 1
    return count

def get_energy(atoms):
    """프레임의 에너지를 반환 (extxyz info field에서 추출)"""
    try:
        return atoms.get_potential_energy()
    except:
        return atoms.info.get('energy', 0.0)

# ==========================================
# 2. 메인 분석 및 추출 루프
# ==========================================
print(f"데이터 추출 시작 (대상 파일: {len(traj_files)}개)\n")

for traj_file in traj_files:
    base_name = os.path.basename(traj_file).replace("_nvt.extxyz", "")
    folder = os.path.basename(os.path.dirname(traj_file))

    print(f"분석 중: {traj_file} ...")
    try:
        traj = read(traj_file, index=':')
    except Exception as e:
        print(f"  -> 파일 읽기 실패: {e}")
        continue
    
    frame_data = []
    for idx, atoms in enumerate(traj):
        b_count = get_bridging_cl_count(atoms)
        energy = get_energy(atoms)
        frame_data.append({
            'index': idx,
            'bridge_count': b_count,
            'energy': energy,
            'atoms': atoms
        })

    # --- 선별 전략 1: Bridging Cl이 많은 순서 (상위 5개) ---
    bridge_candidates = [f for f in frame_data if f['bridge_count'] > 0]
    bridge_candidates.sort(key=lambda x: x['bridge_count'], reverse=True)
    selected_bridge = bridge_candidates[:MAX_BRIDGE_SAMPLES]

    # --- 선별 전략 2: 에너지가 낮은 순서 (상위 5개) ---
    energy_sorted = sorted(frame_data, key=lambda x: x['energy'])
    selected_low_energy = energy_sorted[:MAX_LOW_ENERGY_SAMPLES]

    # --- 선별 전략 3: 에너지가 높은 순서 (상위 5개) ---
    selected_high_energy = energy_sorted[-MAX_HIGH_ENERGY_SAMPLES:]
    selected_high_energy.reverse() # 가장 높은 에너지가 먼저 오도록 정렬

    # --- 파일 저장 루프 ---
    to_save = [
        (selected_bridge, "bridge"),
        (selected_low_energy, "low_energy"),
        (selected_high_energy, "high_energy")
    ]

    for dataset, feature_name in to_save:
        for item in dataset:
            f_idx = item['index']
            
            # 특징 이름 생성 및 폴더명/파일명 지정
            if feature_name == "bridge":
                val = item['bridge_count']
                feature_tag = f"f{f_idx}_bridge{int(val)}"
            elif feature_name == "low_energy":
                val = item['energy']
                feature_tag = f"f{f_idx}_lowEnergy_{val:.2f}"
            elif feature_name == "high_energy":
                val = item['energy']
                feature_tag = f"f{f_idx}_highEnergy_{val:.2f}"

            # 최종 경로 설정: finetuning_dataset/structure_030_c1/bridge/f120_bridge3/f120_bridge3.vasp
            save_path = os.path.join(MAIN_OUTPUT_DIR, base_name, feature_name, feature_tag)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            file_name = f"{feature_tag}.vasp"
            write(os.path.join(save_path, file_name), item['atoms'], format='vasp', vasp5=True, sort=True)

    print(f"  -> 추출 완료: Bridging({len(selected_bridge)}), 저에너지({len(selected_low_energy)}), 고에너지({len(selected_high_energy)})")

print(f"\n모든 작업이 완료되었습니다. 결과 폴더: {MAIN_OUTPUT_DIR}")