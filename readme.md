# Hands-On Novel HPC Workshop

Welcome to the Hands-On Novel HPC Workshop! This repository contains the materials and guides for transitioning your workflows from local, serial execution to parallel, large-scale execution on the Novel HPC cluster.

## 1. Why use HPC?

Moving to a High-Performance Computing (HPC) environment offers two primary benefits:
*   **Exploration (Breadth):** Individual compute forces sequential experimentation. HPC allows for simultaneous exploration of the solution space. Fail faster to succeed sooner.
*   **Scale of Experiments (Depth):** Access to High-Mem GPUs (like the RTX A6000 48GB or A100 80GB) allows for massive batch sizes. This means faster training, stabilized gradients, and the ability to train models that physically cannot fit on local hardware.

## 2. HPC Tooling

*   **Reproducibility with Enroot:** Installing bespoke libraries on shared infrastructure is a nightmare. We use Enroot to run Docker container images without root privileges, ensuring your environment is perfectly reproducible.

    **Enroot Basic Workflow:**
    ```bash
    # 1. Import
    enroot import docker://pytorch/pytorch:latest
    # 2. Create
    enroot create -n my-env pytorch_demo.sqsh
    # 3. Run
    srun --time=03:00:00 --job-name=ttytest --gres=gpu:1 --container-image=./pytorch_demo.sqsh --container-mount-home --pty bash
    ```

*   **Managing Chaos with Tracking:** When running parallel jobs, terminal output is not enough. We highly recommend using Experiment Tracking (e.g., ClearML, W&B, Neptune) to automatically log environments, git commits, code diffs, and live scalar variables (Loss, Accuracy).

## 3. Cluster Access & SSH Keys

You must be on the university network ([e.g. through the Cisco VPN](https://remote.lincoln.ac.uk/)) to access the cluster.

### Basic Access & Testing
```bash
# 0. You should be on the university network
ping -c5 login.novel.hpc

# 1. SSH in (use 10.10.33.11 or 10.10.33.12 if the hostname fails)
ssh <first_initial><lastname>@login.novel.hpc

# 2. Check slurm
sinfo

# 3. Run a test
srun --time=00:00:05 --job-name=testjobg --partition=gpu --gres=gpu:1 nvidia-smi
# or
srun --time=00:00:05 --job-name=testjobc --partition=cpu lscpu
```

### SSH Key Generation & Config
It is highly recommended (and required for VSCode) to generate per-device SSH keys for a minimal attack surface. Run these commands on **YOUR device**:

```bash
# 0. CD to .ssh
cd ~/.ssh/

# 1. Make a key
ssh-keygen -t rsa -b 4096 -f "id_novelfrom${HOSTNAME}_rsa"

# 2. Check permissions in ~/.ssh/
ls -lah

# 3. Copy your key to Novel
ssh-copy-id -i ~/.ssh/id_novelfrom${HOSTNAME}_rsa.pub <username>@login.novel.hpc

# 4. Make or append to your .ssh config
cat <<EOF >> ~/.ssh/config && chmod 600 ~/.ssh/config
Host novel
    HostName login.novel.hpc
    User <username>
    IdentitiesOnly yes
    IdentityFile ~/.ssh/id_novelfrom${HOSTNAME}_rsa
EOF

# 5. Give it a whirl!
ssh novel
```

## 4. Slurm & Quality of Life Utilities

**Slurm** is our workload manager and job scheduler. Key commands include:
*   `sbatch`: Submit a non-interactive script to run in the background.
*   `srun`: Run an interactive (blocking) job.
*   `squeue` / `sinfo`: View job queues and node statuses.
*   `scancel <jobid>`: Terminate a job.

### Installing the QoL Utilities

Native Slurm commands can be verbose. Run this snippet on the **login node** to install wrapper scripts that streamline monitoring and interactive development:

```bash
mkdir -p ~/.local/bin
cd ~/.local/bin
for cmd in lsgpu lsjob erun epy epip; do
    curl -sL "https://s.vnet.tel/$cmd" -o "$cmd"
    chmod +x "$cmd"
done
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

*   `lsgpu` / `lsjob`: Colorized, auto-resizing tables for viewing node resources and current jobs.
*   `erun`: Instantly requests a GPU and drops you into a bash shell inside a PyTorch Enroot container.
*   `epy` / `epip`: Safely run Python or pip inside your virtual environment on a GPU compute node (avoiding login node bottlenecks).

## 5. Workshop Demo: OCR Training

We will train a tiny, single-character OCR model on the EMNIST dataset. The script exports an `.onnx` file that can be dragged into our [Web App (ocrdemo.vnet.tel)](https://ocrdemo.vnet.tel/) for qualitative evaluation. The web app hosted there is also in this repo at `ocr_demo_app.html`. You need to use sftp, or vscode to download the `onnx` file from Novel!

### Setup Environment

```bash
# 1. Clone the repository
git clone https://github.com/LCAS/novel_hpc_workshop.git
cd novel_hpc_workshop

# 2. Create the virtual environment via a compute node using our utility
epy -m venv --copies --system-site-packages .venv

# 3. Activate the environment on the login node
source .venv/bin/activate

# 4. Install requirements safely via a compute node
epip install -r requirements.txt
```

### Addressing Bottlenecks

HPC nodes have immense GPU power but can be bottlenecked by CPU data-loading and image augmentation.
1. I've already ran `python generate_superfast_data.py` (via `epy` or an `sbatch` job) to pre-calculate augmented variants of the EMNIST dataset and save them to RAM-friendly `.npz` files. The output is stored in `/home/shared/air/datasets/emnist/`
2. Train the model using `train_ocr.py`. It includes an HPC guardrail to prevent you from accidentally training on the login node, and supports `--loader normal`, `--loader fast`, and `--loader superfast` to demonstrate (CPU-bound) data loading bottlenecks in this tiny dataset.

### `train.sh`

This file in your repository is for submitting the training script to the Slurm queue. It automatically mounts the required Enroot container and executes the training script using the optimized "superfast" data loader.

```bash
#!/bin/bash
#SBATCH --job-name=ocr_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=registered@email.address
#SBATCH --time=00:15:00 # Using the default queue, jobs cannot exceed 24 hours
# Or
# #SBATCH --qos=long
# #SBATCH --time=2-00:00:00 # i.e. 2 days, you can go up to 7 with long QOS
#SBATCH --output=logs/ocr_train_%j.log
#SBATCH --error=logs/ocr_train_%j.err
#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Enroot container configurations
#SBATCH --container-image=/home/shared/air/enroot-images/pytorch_n_friends.sqsh
#SBATCH --container-mount-home
#SBATCH --container-mounts=/home/shared:/home/shared:ro

# Unload modules that could interfere
module purge

# 1. Navigate to the working directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# 2. Run the training script (sourcing the venv and running python in the same container step)
echo "Starting training job in $PWD..."
srun bash -c "cd $SLURM_SUBMIT_DIR && source .venv/bin/activate && python train_ocr.py --loader superfast --batch-size 256"
```

**To submit the job and monitor live logs:**
```bash
# 1. Submit the script
sbatch train.sh

# 2. Check its status in the queue
lsjob -w

# 3. Watch the live output logs (Ctrl+C to exit)
tail -f logs/*
```

**Checking Job Resource Utilization:**
It is important to ensure you aren't requesting far too much (or too little) memory, CPU, or GPU resources for your jobs. You can use the `sperf` utility followed by your Job ID (e.g., `sperf 9840`) to check the exact resource utilization of a running or recently completed job:

```bash
sperf 9840
```
*Example Output:*
```text
RESOURCE UTILISATION REPORT
Job ID                            : 9840
Cluster name                      : hpc-novel
Running time                      : 00:01:32
Allocated nodes                   : 1
Allocated memory                  : 64.000 GB
Used memory                       : 16.365 GB
Memory utilisation                : 25.57%
Allocated physical CPU cores      : 4
Simultaneous multithreading (SMT) : enabled
CPU utilisation (normalised)      : 5.10%
Allocated GPUs                    : 1
GPU utilisation (normalised)      : 32.64%
Used GPU memory                   : 402.000 MB
------ unformatted metrics ------
Running time (seconds)            : 92
Allocated memory (MB)             : 65536
Used memory (MB)                  : 16758
Used GPU memory (MB)              : 402
```