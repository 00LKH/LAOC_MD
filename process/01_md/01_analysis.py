## Analysis ## Analysis ## Analysis ## Analysis ## Analysis ## Analysis ## Analysis ## Analysis 
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii, atomic_numbers

# ==========================================
# 1. 설정 및 파라미터 
# ==========================================
# MD 계산에서 사용한 반경 및 Cutoff
r_al = covalent_radii[atomic_numbers['Al']]
r_o  = covalent_radii[atomic_numbers['O']]
r_cl = covalent_radii[atomic_numbers['Cl']]

AL_O_BOND_CUTOFF = r_al + r_o + 0.5    # 약 2.42 Å (파괴 기준선)
AL_O_HOOKEAN_RT  = AL_O_BOND_CUTOFF + 0.2  # 약 2.62 Å (복원력 시작선)
AL_CL_CUTOFF     = r_al + r_cl + 0.3   # 약 2.58 Å (Cl 결합 탐색선)

BASE_DIR = "mlmd_extxyz"

# 분석 대상 폴더 탐색
target_folders = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f)) and f.startswith("structure_")])

# 데이터 저장을 위한 변수
composition_data = []
md_plot_data = {}

print(f"총 {len(target_folders)}개의 구조에 대해 종합 분석을 시작합니다...\n" + "="*60)

# ==========================================
# 2. 메인 분석 루프
# ==========================================
for folder in target_folders:
    base_name = folder.replace(".extxyz", "")
    traj_file = os.path.join(BASE_DIR, folder, f"{base_name}_nvt.extxyz")
    
    # x 값 추출 (예: structure_030_c1 -> 0.3)
    try:
        x_val = float(base_name.split('_')[1]) / 100.0
    except:
        x_val = 0.0
    
    if not os.path.exists(traj_file):
        continue
        
    print(f"분석 진행 중: {folder} (x = {x_val:.2f})")
    traj = read(traj_file, index=':')
    initial_atoms = traj[0]
    
    chain_ids = initial_atoms.get_array('chain_id')
    al_roles = initial_atoms.get_array('al_role')
    
    # ----------------------------------------------------
    # [목표 1] 초기 구조 결함 분포 (0b, 1b, 2b) 분석
    # ----------------------------------------------------
    cl_i, al_j = neighbor_list('ij', initial_atoms, {('Cl', 'Al'): AL_CL_CUTOFF})
    mid_al_indices = [i for i, role in enumerate(al_roles) if role == 1]
    
    counts = {0: 0, 1: 0, 2: 0} # 2 Cls(0b), 1 Cl(1b), 0 Cl(2b)
    
    for al_idx in mid_al_indices:
        # 이 중앙 Al에 결합된 Cl의 개수 파악
        attached_cls = sum(1 for a in al_j if a == al_idx)
        if attached_cls == 2:
            counts[0] += 1 # 0 broken
        elif attached_cls == 1:
            counts[1] += 1 # 1 broken
        else:
            counts[2] += 1 # 2 broken
            
    total_mid_al = len(mid_al_indices)
    composition_data.append({
        'x_val': x_val,
        'Folder': folder,
        '0b_Count': counts[0],
        '1b_Count': counts[1],
        '2b_Count': counts[2],
        '0b_Ratio(%)': (counts[0]/total_mid_al)*100 if total_mid_al else 0,
        '1b_Ratio(%)': (counts[1]/total_mid_al)*100 if total_mid_al else 0,
        '2b_Ratio(%)': (counts[2]/total_mid_al)*100 if total_mid_al else 0,
    })

    # ----------------------------------------------------
    # [목표 2 & 3] 체인 유지 검증 및 브릿징 품질 분석
    # ----------------------------------------------------
    # 체인 내 초기 Al-O 쌍 식별
    ini_i, ini_j = neighbor_list('ij', initial_atoms, AL_O_BOND_CUTOFF)
    chain_bonds = [(i, j) for i, j in zip(ini_i, ini_j) if i < j and chain_ids[i] == chain_ids[j] and chain_ids[i] > 0 and {initial_atoms[i].symbol, initial_atoms[j].symbol} == {'Al', 'O'}]

    frames, max_dists, bridge_counts, avg_deltas = [], [], [], []
    best_bridge_frames = [] # (품질 점수, 프레임 번호, atoms)

    for f_idx, atoms in enumerate(traj):
        frames.append(f_idx)
        
        # 1. 최대 Al-O 거리 (Hookean)
        max_dist = max([atoms.get_distance(i, j, mic=True) for i, j in chain_bonds]) if chain_bonds else 0
        max_dists.append(max_dist)
        
        # 2. 브릿징 품질 분석
        c_i, a_j = neighbor_list('ij', atoms, {('Cl', 'Al'): AL_CL_CUTOFF})
        cl_to_al = {}
        for c, a in zip(c_i, a_j):
            if atoms[c].symbol == 'Cl':
                cl_to_al.setdefault(c, []).append(a)
                
        frame_bridge_count = 0
        frame_deltas = []
        best_delta_in_frame = float('inf')
        best_mu_in_frame = float('inf')

        for cl_idx, al_indices in cl_to_al.items():
            if len(al_indices) == 2:
                a1, a2 = al_indices
                c1, c2 = chain_ids[a1], chain_ids[a2]
                r1, r2 = al_roles[a1], al_roles[a2]
                
                # 동일 체인 내에서 중앙 Al(1)과 양끝 Al(2)을 잇는 경우
                if c1 == c2 and c1 > 0 and {r1, r2} == {1, 2}:
                    frame_bridge_count += 1
                    d1 = atoms.get_distance(cl_idx, a1, mic=True)
                    d2 = atoms.get_distance(cl_idx, a2, mic=True)
                    
                    delta = abs(d1 - d2)   # 분산 대용 (비대칭도)
                    mu = (d1 + d2) / 2.0   # 평균 결합 길이
                    
                    frame_deltas.append(delta)
                    
                    # 가장 대칭적(delta 낮음)이고 타이트한(mu 낮음) 결합 추적
                    if delta < best_delta_in_frame:
                        best_delta_in_frame = delta
                        best_mu_in_frame = mu
                        
        bridge_counts.append(frame_bridge_count)
        avg_deltas.append(np.mean(frame_deltas) if frame_deltas else np.nan)
        
        if frame_bridge_count > 0:
            # 품질 점수: 비대칭도가 낮을수록(0에 가까울수록), 결합길이가 짧을수록 좋음
            score = best_delta_in_frame + (0.1 * best_mu_in_frame)
            best_bridge_frames.append((score, best_delta_in_frame, best_mu_in_frame, f_idx, atoms))

    md_plot_data[x_val] = {
        'frames': frames, 'max_dists': max_dists, 
        'bridge_counts': bridge_counts, 'avg_deltas': avg_deltas
    }

# ==========================================
# 3. 데이터 저장 및 시각화 (Figure 1: 결함 비율)
# ==========================================
print("\n[완료] 데이터 분석 완료. 시각화 자료를 생성합니다...")

# CSV 저장
df_comp = pd.DataFrame(composition_data)

# 데이터가 비어있지 않은 경우에만 정렬 실행
if not df_comp.empty:
    df_comp = df_comp.sort_values(by='x_val')

df_comp.to_csv("composition_summary.csv", index=False)

# Figure 1: 누적 막대 그래프
with plt.style.context(["science", "notebook"]):
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    x_labels = [f"LAOC_{x:.1f}B" for x in df_comp['x_val']]
    bar_0b = df_comp['0b_Ratio(%)'].values
    bar_1b = df_comp['1b_Ratio(%)'].values
    bar_2b = df_comp['2b_Ratio(%)'].values

    ax1.bar(x_labels, bar_0b, label='0b (0 Cl removed)', color='C0')
    ax1.bar(x_labels, bar_1b, bottom=bar_0b, label='1b (1 Cl removed)', color='C1')
    ax1.bar(x_labels, bar_2b, bottom=bar_0b + bar_1b, label='2b (2 Cl removed)', color='C2')

    # ax1.set_xlabel("Global Composition (x)")
    ax1.set_ylabel("0b : 1b : 2b ratio (%)")
    
    # [수정됨] 범례를 그래프 안쪽 우측 상단에 배치하고, 배경을 반투명(alpha=0.7)하게 설정
    ax1.legend(loc='upper right', frameon=True, framealpha=0.7, edgecolor='white')
    
    plt.tight_layout()
    plt.show()
    fig1.savefig("Figure1_Defect_Ratio.png", dpi=300)

# ==========================================
# 4. 데이터 시각화 (Figure 2: MD 궤적 Subplot)
# ==========================================
num_plots = len(md_plot_data)
if num_plots > 0:
    with plt.style.context(["science", "notebook"]):
        # [수정됨] 범례가 들어갈 공간을 확보하기 위해 가로 길이를 16으로 약간 늘림
        fig2, axs = plt.subplots(1, 2, figsize=(16, 5))

        sorted_x_vals = sorted(md_plot_data.keys())

        for idx, x_val in enumerate(sorted_x_vals):
            data = md_plot_data[x_val]
            fr = data['frames']
            
            # [수정됨] 라벨 이름 변경 (예: x=0.2 -> LAOC_0.2b)
            # ':g' 포맷을 사용하면 0.20의 불필요한 0을 지워 0.2로 깔끔하게 출력합니다.
            label_name = f'LAOC_{x_val:g}B' 

            # Col 0, Col 1 플롯 그리기
            axs[0].plot(fr, data['max_dists'], label=label_name, linewidth=4)
            axs[1].plot(fr, data['bridge_counts'], label=label_name, linewidth=4)

        # Cutoff 라인은 플롯에 한 번만 추가
        axs[0].axhline(y=AL_O_BOND_CUTOFF, color='black', linestyle='--', label='Bond Cutoff')#, alpha=0.5
        axs[0].axhline(y=AL_O_HOOKEAN_RT, color='blue', linestyle=':', label='Hookean')#, alpha=0.5

        # 첫 번째 그래프 축 설정
        # axs[0].set_title("1. Chain Stability (Max Al-O)")
        axs[0].set_ylabel("Max Al-O (Å)")
        axs[0].set_xlabel("Time")
        # [수정됨] 범례를 그래프 우측 바깥으로 이동
        axs[0].legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True) 

        # 두 번째 그래프 축 설정
        # axs[1].set_title("2. Intra-Chain Bridging Cl Count")
        axs[1].set_ylabel("# of Bridging Cl")
        axs[1].set_xlabel("Time")
        # [수정됨] 범례를 그래프 우측 바깥으로 이동
        axs[1].legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)

        plt.tight_layout()
        plt.show()
        
        # [수정됨] 바깥으로 빠진 범례가 이미지 저장 시 잘리지 않도록 bbox_inches='tight' 추가
        fig2.savefig("Figure2_MD_Dynamics_Grid.png", dpi=300, bbox_inches='tight')

print("모든 작업이 완료되었습니다!")
print("- 조성 데이터: composition_summary.csv / Figure1_Defect_Ratio.png")
print("- MD 다이내믹스 그래프: Figure2_MD_Dynamics_Grid.png")