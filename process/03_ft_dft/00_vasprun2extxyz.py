import ase.io
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import sys

# 검색할 대상 최상위 디렉토리 (기존 경로를 유지하되 필요에 따라 변경 가능)
# base_dir = "/home/kyunghun/01_halide/01_test"
# 현재 디렉토리부터 탐색하고 싶다면 아래와 같이 변경하세요:
base_dir = "./extracted_xmls"

all_frames = []
print(f"Finding and reading all vasprun.xml files in {base_dir} and its subdirectories...")

# rglob을 사용하여 하위 디렉토리를 포함한 모든 vasprun.xml 검색
for xml_file in Path(base_dir).rglob("vasprun.xml"):
    print(f"Reading {xml_file}...")
    try:
        frames = ase.io.read(xml_file, index=":")
        all_frames.extend(frames)
    except Exception as e:
        print(f"Error reading {xml_file}: {e}")

print(f"Total frames read: {len(all_frames)}")

if not all_frames:
    print("No frames found. Exiting...")
    sys.exit()

# 2. 데이터 분할 (8:1:1 비율 예시)
train_frames, valid_frames = train_test_split(all_frames, test_size=0.2, random_state=42)
# valid_frames, test_frames = train_test_split(temp_frames, test_size=0.5, random_state=42)

def save_extxyz(frames, filename):
    valid_frames = []
    
    for atoms in frames:
        try:
            # 에너지가 없는 프레임(불완전한 계산 결과)을 걸러내기 위해 시도
            atoms.info["REF_energy"] = atoms.get_potential_energy()
            atoms.arrays["REF_forces"] = atoms.get_forces()
            
            # stress는 VASP 세팅에 따라 없을 수도 있으므로 별도로 예외 처리
            try:
                atoms.info["REF_stress"] = atoms.get_stress()
            except Exception:
                pass
            
            # 모든 필수 정보가 있는 프레임만 리스트에 추가
            valid_frames.append(atoms)
                
        except Exception as e:
            # 계산기(calculator)가 없거나 property가 없는 프레임은 생략
            pass
            
    ase.io.write(filename, valid_frames, format="extxyz")
    print(f"Saved {len(valid_frames)} frames to {filename} (skipped {len(frames) - len(valid_frames)} invalid frames)")


save_extxyz(train_frames, "train.extxyz")
save_extxyz(valid_frames, "valid.extxyz")
# save_extxyz(test_frames, "test.extxyz")
