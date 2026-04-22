import numpy as np
import random
from ase.io import read, write
import os
os.makedirs('mlmd_vasp', exist_ok=True) # 폴더가 없으면 생성
os.makedirs('../01_md/mlmd_extxyz', exist_ok=True) # 폴더가 없으면 생성
# ---------------------------------------------------------
# 설정 및 준비
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1. 태그가 완료된 초기 구조 읽기 (Cl에 1, 2 태그가 부여된 상태)
atoms_base = read('Al3O2Cl8_tagged.extxyz') 

# 2. 지정된 x 값 및 생성할 후보 개수
x_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
num_candidates = 1

# 3. Formula Unit (Z) 계산 (Al3O2Cl8 기준)
num_al = sum(1 for atom in atoms_base if atom.symbol == 'Al')
Z = num_al / 3.0

# 4. 태그가 1로 지정된 Cl 원자들의 인덱스 수집
# tagged_cl_indices = [atom.index for atom in atoms_base if atom.tag == 1 and atom.symbol == 'Cl']

cl_tag_array = atoms_base.get_array('cl_tag')

tagged_cl_indices = [
    i for i, val in enumerate(cl_tag_array) 
    if val == 1 and atoms_base[i].symbol == 'Cl'
]

print(f"총 Formula Units (Z): {Z}")
print(f"제거 후보(태그 1) Cl 원자 총 개수: {len(tagged_cl_indices)}\n")
print("=" * 60)

# ---------------------------------------------------------
# Li 배치 함수
# ---------------------------------------------------------
def generate_li_positions(atoms, num_li_to_add, min_dist=1.8, max_attempts=5000):
    cell = atoms.get_cell()
    new_li_positions = []
    
    attempts = 0
    while len(new_li_positions) < num_li_to_add and attempts < max_attempts:
        attempts += 1
        
        # 셀 내부 무작위 좌표 생성
        frac_coords = np.random.rand(3)
        cart_coords = np.dot(frac_coords, cell)
        
        # 거리 계산을 위한 임시 구조체
        temp_atoms = atoms.copy()
        temp_atoms.append('Li')
        temp_atoms.positions[-1] = cart_coords
        
        # 추가하려는 위치와 기존 원자들 간의 최단 거리 측정 (주기적 경계조건 적용)
        distances = temp_atoms.get_distances(-1, range(len(temp_atoms)-1), mic=True)
        
        # 모든 거리가 min_dist 이상이면 승인
        if np.all(distances >= min_dist):
            new_li_positions.append(cart_coords)
            atoms.append('Li')
            atoms.positions[-1] = cart_coords
            
    if len(new_li_positions) < num_li_to_add:
        print(f"  [경고] {max_attempts}번 시도했으나 {num_li_to_add}개의 Li을 모두 배치하지 못했습니다. (현재 {len(new_li_positions)}개)")
        
    return atoms

# ---------------------------------------------------------
# 메인 생성 루프
# ---------------------------------------------------------
for x in x_values:
    # 소수점 오차 방지를 위해 round 사용
    num_cl_to_remove = round(Z * x)
    num_li_to_add = round(Z * (3 - x))
    file_suffix = int(x * 100)
    
    print(f"▶ x = {x:<4} (파일명: {file_suffix:03d}) | Cl 제거: {num_cl_to_remove}개 | Li 추가: {num_li_to_add}개")
    
    if num_cl_to_remove > len(tagged_cl_indices):
        print(f"  [오류] 태그된 Cl 원자 수가 부족하여 건너뜁니다.\n")
        continue

    # 후보 구조 3개 생성
    for i in range(num_candidates):
        # 1) 원본 구조에서 출발
        atoms_current = atoms_base.copy()
        
        # 2) Cl 제거 (x 비율만큼)
        # 루프마다 random.sample이 호출되므로 3개의 후보 구조는 서로 다른 Cl 빈자리를 가질 수 있습니다.
        # 동일한 x값 내에서 Cl 빈자리는 고정하고 Li 위치만 바꾸고 싶다면, 
        # 이 단계(Cl 제거)를 i 루프 밖으로 빼시면 됩니다.
        current_cl_indices = random.sample(tagged_cl_indices, num_cl_to_remove)
        del atoms_current[current_cl_indices]
        
        # 3) Li 배치
        atoms_final = generate_li_positions(atoms_current, num_li_to_add, min_dist=1.8)
        
        # 4) 파일 저장 (VASP 및 XYZ 동시 저장)
        # 예: POSCAR_030_c1.vasp / structure_030_c1.xyz
        structure_name = f'structure_{file_suffix:03d}_c{i+1}'
        vasp_filename = f'mlmd_vasp/POSCAR_{file_suffix:03d}_c{i+1}.vasp'
        
        xyz_dir = f'../01_md/mlmd_extxyz/{structure_name}'
        os.makedirs(xyz_dir, exist_ok=True)
        xyz_filename = f'{xyz_dir}/{structure_name}.extxyz'
        
        write(vasp_filename, atoms_final, format='vasp', sort=True)
        write(xyz_filename, atoms_final, format='extxyz')
        
        print(f"  - 후보 {i+1} 생성 완료: {vasp_filename}")
        
    print("-" * 60)

print("모든 구조 생성이 완료되었습니다!")