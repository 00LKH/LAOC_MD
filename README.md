# LAOC_MD
conda create -n mace python=3.10 -y \n
conda activate mace \n
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \n
pip install mace-torch cuequivariance-torch agox pymatgen scienceplots ipykernel cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu11 ipykernel ase scipp kinisi \n
