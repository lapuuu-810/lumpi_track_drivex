# Checkpoint manifest

## Shipped with the package (UCF specialists, copied from the UCF project)
- ucf_car65a_e8.pth   (baseline65a_car_only_mild_e8/checkpoint_epoch_8.pth)
- ucf_7cls55a_e2.pth  (baseline55a_morph_mild_from_b1e12_e2/checkpoint_epoch_2.pth)
- ucf_bus125a_e8.pth  (baseline125a_bus_only_realsize_synth_e8/checkpoint_epoch_8.pth)

## To restore (DSVT + rldm2k; originally under the 121/140 workspaces, which were
## cleaned/decommissioned)
Place the following files into this directory and `run_detection_repro.sh`
(without GPU, final mode) reproduces the final answer; `--mode full` additionally
runs the ten-model inference chain.

| Expected file name | Original path (under the OpenPCDet output directory) |
|---|---|
| dsvt_final_epoch30.pth | configs/dsvt_vfield_g0s_lumpi_final/..._8gpu/ckpt/checkpoint_epoch_30.pth |
| dsvt_final_epoch32.pth | same, epoch_32 |
| dsvt_clean_epoch34.pth | configs/dsvt_vfield_g0s_lumpi_clean/..._8gpu/ckpt/checkpoint_epoch_34.pth |
| dsvt_pedalign_epoch40.pth | configs/dsvt_vfield_g0s_lumpi_pedalign/..._2gpu_b16/ckpt/checkpoint_epoch_40.pth |
| dsvt_v4_epoch36/38/40.pth | configs/dsvt_vfield_g0s_lumpi_v4/..._8gpu/ckpt/checkpoint_epoch_36/38/40.pth |
| dsvt_rldm2k_epoch40.pth | configs/dsvt_vfield_g0s_lumpi_d50k_rldm2k/..._2gpu_b16/ckpt/checkpoint_epoch_40.pth |

Restoration enables the full mode of the original pipeline; without the weights
the default final mode still produces the exact online answer.
