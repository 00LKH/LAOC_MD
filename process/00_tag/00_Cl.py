from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN
import numpy as np

atoms = read("Al3O2Cl8.cif")
atoms = atoms*(1,1,2)
atoms.write("Al3O2Cl8.extxyz", format="extxyz")

# 1. 설정 및 파일 읽기
input_file = "Al3O2Cl8.extxyz"  # 원본 구조 파일
output_file = "Al3O2Cl8_tagged.extxyz"

atoms = read(input_file)
structure = AseAtomsAdaptor.get_structure(atoms)
cnn = CrystalNN()

# 태그를 저장할 별도의 numpy 배열 초기화
cl_tags = np.zeros(len(atoms), dtype=int)
chain_tags = np.zeros(len(atoms), dtype=int)

print("태깅 작업을 시작합니다...")

# ------------------------------------------------------------
# 2. [작업 A] 특정 Cl 원자 태깅 (Al에 2개 붙은 Cl 찾기)
# ------------------------------------------------------------
for idx, site in enumerate(structure):
    if site.species_string == 'Al':
        nn_info = cnn.get_nn_info(structure, idx)
        cl_neighbors = [n['site_index'] for n in nn_info if n['site'].species_string == 'Cl']
        
        # Al에 Cl이 정확히 2개 붙어있는 경우, 해당 Cl들에 태그 1 부여
        if len(cl_neighbors) == 2:
            cl_tags[cl_neighbors[0]] = 1
            cl_tags[cl_neighbors[1]] = 1

print("- Cl 태깅 완료")

# ------------------------------------------------------------
# 3. [작업 B] Al-O-Al-O-Al 5원자 체인 태깅
# ------------------------------------------------------------
chain_id_counter = 1
intact_count = 0

for idx, site in enumerate(structure):
    if site.species_string == 'Al':
        # 이미 체인 ID가 부여된 Al은 건너뜀
        if chain_tags[idx] != 0:
            continue
            
        nn_info = cnn.get_nn_info(structure, idx)
        o_neighbors = [n['site_index'] for n in nn_info if n['site'].species_string == 'O']
        
        # 중심 Al(Mid-Al) 후보 찾기 (O가 2개인 Al)
        if len(o_neighbors) == 2:
            o1_idx, o2_idx = o_neighbors
            
            # 산소와 연결된 다른 Al(End-Al) 탐색
            o1_nn = cnn.get_nn_info(structure, o1_idx)
            o2_nn = cnn.get_nn_info(structure, o2_idx)
            
            end1_candidates = [n['site_index'] for n in o1_nn if n['site'].species_string == 'Al' and n['site_index'] != idx]
            end2_candidates = [n['site_index'] for n in o2_nn if n['site'].species_string == 'Al' and n['site_index'] != idx]
            
            if end1_candidates and end2_candidates:
                e1_idx = end1_candidates[0]
                e2_idx = end2_candidates[0]
                
                # 5개 원자에 고유 체인 ID 부여
                chain_indices = [idx, o1_idx, o2_idx, e1_idx, e2_idx]
                for c_idx in chain_indices:
                    chain_tags[c_idx] = chain_id_counter
                
                chain_id_counter += 1
                intact_count += 1

print(f"- 체인 태깅 완료 (총 {intact_count}개 식별)")

# ------------------------------------------------------------
# 4. 결과 저장 (ASE Atoms 객체에 데이터 배열 추가)
# ------------------------------------------------------------
# tags 속성 대신 사용자 정의 이름을 사용하여 중복 방지
atoms.set_array('cl_tag', cl_tags)
atoms.set_array('chain_id', chain_tags)

# (참고) 기존 tags 속성에도 체인 정보를 넣어두고 싶다면:
atoms.set_tags(chain_tags)

atoms.write(output_file, format="extxyz")
print("-" * 60)
print(f"최종 파일이 저장되었습니다: {output_file}")