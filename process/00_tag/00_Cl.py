from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN
import numpy as np

# ------------------------------------------------------------
# 1. 초기 파일 읽기 및 슈퍼셀 생성 (기존 유지)
# ------------------------------------------------------------
atoms = read("Al3O2Cl8.cif")
atoms = atoms * (1, 1, 2)
atoms.write("Al3O2Cl8.extxyz", format="extxyz")

input_file = "Al3O2Cl8.extxyz"  
output_file = "Al3O2Cl8_tagged.extxyz"

print(f"[{input_file}] 파일을 읽고 분석을 시작합니다...\n" + "-"*60)
atoms = read(input_file)

# Pymatgen 구조 변환 및 배위 환경 분석기 설정
structure = AseAtomsAdaptor.get_structure(atoms)
cnn = CrystalNN()

# ------------------------------------------------------------
# 2. 독립적인 태그 저장을 위한 Numpy 배열 초기화
# ------------------------------------------------------------
num_atoms = len(atoms)
chain_ids = np.zeros(num_atoms, dtype=int) # 체인 번호 (1, 2, 3...)
al_roles = np.zeros(num_atoms, dtype=int)  # 1: 가운데 Al (Mid), 2: 양끝 Al (End)
cl_tags = np.zeros(num_atoms, dtype=int)   # 1: 가운데 Al에 붙은 Cl

chain_counter = 1
tagged_cl_count = 0

# ------------------------------------------------------------
# 3. 체인 탐색 및 태깅 (Al-O-Al-O-Al)
# ------------------------------------------------------------
for idx, site in enumerate(structure):
    if site.species_string == 'Al':
        # 이미 체인에 소속된 Al이면 건너뜀
        if chain_ids[idx] != 0:
            continue
            
        # 현재 Al의 이웃 원자들 분석
        nn_info = cnn.get_nn_info(structure, idx)
        o_neighbors = [n for n in nn_info if n['site'].species_string == 'O']
        
        # [조건 1] 산소(O)가 정확히 2개 붙어있다면 '가운데 Al (Mid-Al)' 후보
        if len(o_neighbors) == 2:
            o1_idx = o_neighbors[0]['site_index']
            o2_idx = o_neighbors[1]['site_index']
            
            # 해당 산소들이 이미 다른 체인에 속해있는지 확인
            if chain_ids[o1_idx] != 0 or chain_ids[o2_idx] != 0:
                continue
                
            # [조건 2] 각 산소에 연결된 다른 Al ('양끝 Al / End-Al') 탐색
            o1_nn = cnn.get_nn_info(structure, o1_idx)
            o2_nn = cnn.get_nn_info(structure, o2_idx)
            
            end1_cands = [n['site_index'] for n in o1_nn if n['site'].species_string == 'Al' and n['site_index'] != idx]
            end2_cands = [n['site_index'] for n in o2_nn if n['site'].species_string == 'Al' and n['site_index'] != idx]
            
            # 양끝 Al이 모두 정상적으로 존재할 경우 최종 체인 확정
            if end1_cands and end2_cands:
                end1_idx = end1_cands[0]
                end2_idx = end2_cands[0]
                
                # --- A. 체인 ID 부여 (5원자 모두 동일 번호) ---
                chain_indices = [idx, o1_idx, o2_idx, end1_idx, end2_idx]
                for c_idx in chain_indices:
                    chain_ids[c_idx] = chain_counter
                    
                # --- B. Al 위치 구분 태그 부여 (al_role 생성 부분) ---
                al_roles[idx] = 1       # 가운데 Al = 1
                al_roles[end1_idx] = 2  # 양끝 Al = 2
                al_roles[end2_idx] = 2  # 양끝 Al = 2
                
                # --- C. 가운데 Al에 결합된 Cl 태깅 ---
                # idx(가운데 Al)의 이웃 정보(nn_info)에서 Cl만 추출
                cl_neighbors = [n['site_index'] for n in nn_info if n['site'].species_string == 'Cl']
                for cl_idx in cl_neighbors:
                    cl_tags[cl_idx] = 1
                    tagged_cl_count += 1
                    
                chain_counter += 1

# ------------------------------------------------------------
# 4. 분석 결과 출력 및 파일 저장 (.extxyz)
# ------------------------------------------------------------
# 생성한 배열들을 ASE 원자 객체에 안전하게 독립 속성으로 추가
atoms.set_array('chain_id', chain_ids)
atoms.set_array('al_role', al_roles)
atoms.set_array('cl_tag', cl_tags)

atoms.write(output_file, format='extxyz')

print(f"▶ 총 식별된 체인 개수: {chain_counter - 1} 개")
print(f"▶ 가운데 Al에 결합되어 태깅된 Cl 총 개수: {tagged_cl_count} 개")
print("-" * 60)
print(f"모든 태깅이 완료되었습니다! 파일 저장: {output_file}")