# NVIDIA L4 Driver Installation (Debian 11)

This guide installs the correct NVIDIA driver (≥525) for an L4 GPU on Debian 11, with optional CUDA and a quick PyTorch CUDA check.

## 1. Update the system
Update package lists and upgrade installed packages to avoid dependency conflicts.
```bash
sudo apt update && sudo apt upgrade -y
```

## 2. Remove old NVIDIA drivers (if any)
Clean out previous driver or CUDA installs to prevent version clashes.
```bash
sudo apt autoremove 'nvidia*' --purge
# If you previously used .run installers:
sudo /usr/bin/nvidia-uninstall                # NVIDIA driver .run
sudo /usr/local/cuda-X.Y/bin/cuda-uninstall   # CUDA .run
```

## 3. Enable non-free repositories
Debian’s proprietary NVIDIA packages require contrib and non-free components.
```bash
sudo apt install software-properties-common -y
sudo add-apt-repository contrib non-free
sudo apt update
```

## 4. Identify the GPU
Confirm that the system sees the L4 GPU and that you are on NVIDIA hardware.

```bash
lspci -nn | egrep -i "3d|display|vga"
# Expect an NVIDIA data center GPU (L4 requires driver 525+).
```

## 5. Install prerequisites
Install tools for HTTPS repositories, DKMS, and key handling.
```bash
sudo apt install -y dirmngr ca-certificates software-properties-common \
    apt-transport-https dkms curl
```

## 6. Import NVIDIA GPG key (Debian 11)
Add NVIDIA’s signing key so APT can verify packages from the CUDA repo.
```bash
curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor \
  | sudo tee /usr/share/keyrings/nvidia-drivers.gpg >/dev/null 2>&1
```

## 7. Add the NVIDIA CUDA repository
Add the CUDA repo to access recent L4-capable drivers (525+), then refresh APT.
```bash
echo 'deb [signed-by=/usr/share/keyrings/nvidia-drivers.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /' \
  | sudo tee /etc/apt/sources.list.d/nvidia-drivers.list

sudo apt update
```

## 8. Ensure matching kernel headers
Install headers for your running kernel so DKMS can build NVIDIA modules.

```bash
uname -r
sudo apt update
sudo apt install -y linux-headers-$(uname -r)
```

## 9. Install the NVIDIA driver (with or without CUDA)
Choose one of the following, depending on whether you need CUDA libraries.
- Driver only (no CUDA toolkit):
    ```bash
    sudo apt install -y nvidia-driver nvidia-smi nvidia-settings
    ```
- Driver + CUDA toolkit:
    ```bash
    sudo apt install -y nvidia-driver cuda nvidia-smi nvidia-settings
    ```
If prompted to remove conflicting packages, accept so the new driver can install cleanly.

Reboot to load the new kernel modules.
```bash
sudo reboot
```

## 10. Verify driver installation
After reboot, confirm that the driver sees the L4 GPU.
```bash
nvidia-smi
# Should show the L4 GPU, driver version (≥525), and CUDA version.
```

## 11. Test CUDA from PyTorch
Check that CUDA is available at the framework level (example with PyTorch).
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expect: True
```