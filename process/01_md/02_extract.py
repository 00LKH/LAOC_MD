## Extract ## Extract ## Extract ## Extract ## Extract ## Extract 
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
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

RUN_DIR = "."
BASE_DIR = "./mlmd_extxyz"
MAIN_OUTPUT_DIR = "../02_mlmd_extract/finetuning_dataset" # 병합된 최상위 저장 폴더

# 추출할 구조 개수 설정
MAX_BRIDGE_SAMPLES = 10
MAX_LOW_ENERGY_SAMPLES = 10
MAX_HIGH_ENERGY_SAMPLES = 10

# 분석할 궤적 파일 재귀적 탐색
traj_files = sorted(glob.glob(os.path.join(BASE_DIR, "**/*_nvt.extxyz"), recursive=True))

# --- 전체 시각화를 위한 설정 추가 ---
n_files = len(traj_files)
ncols = 3
nrows = (n_files + ncols - 1) // ncols  # 11개일 경우 4행 생성

with plt.style.context(["science", "notebook"]):
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes_flat = axes.flatten()

    def get_bridging_cl_count(atoms):
        """현재 프레임에서 Intra-chain Bridging Cl의 개수를 반환 (01_analysis.py 로직 동기화)"""
        # 체인 정보나 역할 정보가 없는 경우를 대비한 예외 처리
        if 'chain_id' not in atoms.arrays or 'al_role' not in atoms.arrays:
            return 0

        chain_ids = atoms.get_array('chain_id')
        al_roles = atoms.get_array('al_role')
        cl_i, al_j = neighbor_list('ij', atoms, {('Cl', 'Al'): AL_CL_CUTOFF})

        cl_to_al = {}
        for c, a in zip(cl_i, al_j):
            if atoms[c].symbol == 'Cl':
                if c not in cl_to_al: cl_to_al[c] = []
                cl_to_al[c].append(a)

        count = 0
        for cl_idx, al_indices in cl_to_al.items():
            if len(al_indices) == 2:
                a1, a2 = al_indices
                c1, c2 = chain_ids[a1], chain_ids[a2]
                r1, r2 = al_roles[a1], al_roles[a2]

                # 동일 체인 내에서 중앙 Al(1)과 양끝 Al(2)을 잇는 경우만 브릿징으로 판별
                if c1 == c2 and c1 > 0 and {r1, r2} == {1, 2}:
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
    print(f"데이터 추출 및 시각화 시작 (대상 파일: {len(traj_files)}개)\n" + "="*60)

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
        all_indices = []
        all_energies = []

        for idx, atoms in enumerate(traj):
            b_count = get_bridging_cl_count(atoms)
            energy = get_energy(atoms)

            frame_data.append({
                'index': idx,
                'bridge_count': b_count,
                'energy': energy,
                'atoms': atoms
            })

            # 플롯팅을 위한 전체 데이터 저장
            all_indices.append(idx)
            all_energies.append(energy)

        # --- 선별 전략 1: Bridging Cl이 많은 순서 ---
        bridge_candidates = [f for f in frame_data if f['bridge_count'] > 0]
        bridge_candidates.sort(key=lambda x: x['bridge_count'], reverse=True)
        selected_bridge = bridge_candidates[:MAX_BRIDGE_SAMPLES]

        # --- 선별 전략 2: 에너지가 낮은 순서 ---
        energy_sorted = sorted(frame_data, key=lambda x: x['energy'])
        selected_low_energy = energy_sorted[:MAX_LOW_ENERGY_SAMPLES]

        # --- 선별 전략 3: 에너지가 높은 순서 ---
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

                if feature_name == "bridge":
                    val = item['bridge_count']
                    feature_tag = f"f{f_idx}_bridge{int(val)}"
                elif feature_name == "low_energy":
                    val = item['energy']
                    feature_tag = f"f{f_idx}_lowEnergy_{val:.2f}"
                elif feature_name == "high_energy":
                    val = item['energy']
                    feature_tag = f"f{f_idx}_highEnergy_{val:.2f}"

                save_path = os.path.join(MAIN_OUTPUT_DIR, base_name, feature_name, feature_tag)
                os.makedirs(save_path, exist_ok=True)

                file_name = f"{feature_tag}.vasp"
                write(os.path.join(save_path, file_name), item['atoms'], format='vasp', vasp5=True, sort=True)

        print(f"  -> 추출 완료: Bridging({len(selected_bridge)}), 저에너지({len(selected_low_energy)}), 고에너지({len(selected_high_energy)})")

        # ==========================================
        # 3. 데이터 시각화 (에너지 Plot 및 추출 지점 표시)
        # ==========================================
        # 각 추출 그룹별 x(index), y(energy) 좌표 분리
        bridge_idx = [item['index'] for item in selected_bridge]
        bridge_ene = [item['energy'] for item in selected_bridge]

        low_idx = [item['index'] for item in selected_low_energy]
        low_ene = [item['energy'] for item in selected_low_energy]

        high_idx = [item['index'] for item in selected_high_energy]
        high_ene = [item['energy'] for item in selected_high_energy]

        # ==========================================
        # 3. 데이터 시각화 (Subplot 각 칸에 그리기)
        # ==========================================
        ax = axes_flat[traj_files.index(traj_file)] # 현재 순서에 맞는 subplot 선택

        # 데이터 분리
        bridge_idx = [item['index'] for item in selected_bridge]
        bridge_ene = [item['energy'] for item in selected_bridge]
        low_idx = [item['index'] for item in selected_low_energy]
        low_ene = [item['energy'] for item in selected_low_energy]
        high_idx = [item['index'] for item in selected_high_energy]
        high_ene = [item['energy'] for item in selected_high_energy]

        # 해당 칸(ax)에 그래프 작성
        ax.plot(all_indices, all_energies, color='gray', zorder=1, linewidth=4) #label='Trajectory',
        ax.scatter(bridge_idx, bridge_ene, color='C0', marker='o', s=40, label='Bridge')
        ax.scatter(low_idx, low_ene, color='C1', marker='o', s=40, label='Low E')
        ax.scatter(high_idx, high_ene, color='C2', marker='o', s=40, label='High E')
        # ax.set_xlabel('Time')

        x_str = base_name.split('_')[1]
        x_val = float(x_str) / 100.0
        custom_label = f"LAOC_{x_val:g}B" # :g는 0.20을 0.2로 깔끔하게 포맷팅해줍니다.

        ax.legend(title=custom_label, title_fontproperties={'weight':'bold'}, loc='best')
        # ax.legend(title=custom_label, title_fontproperties={'weight':'bold', 'size':9}, fontsize=8, loc='best')
        # ax.tick_params(axis='both', which='major', labelsize=8)
        if traj_files.index(traj_file) % ncols == 0:
            ax.set_ylabel('Energy (eV)')

    # 사용하지 않는 빈 서브플롯 숨기기 (11개일 경우 마지막 한 칸)
    for j in range(n_files, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout()
    # 전체 요약 이미지 저장
    summary_plot_path = os.path.join(RUN_DIR, "Figure3_extract.png")
    plt.show()
    plt.savefig(summary_plot_path, dpi=300)
    print(f"\n[체크] 모든 구조의 에너지 분포가 {summary_plot_path}에 통합 저장되었습니다.")