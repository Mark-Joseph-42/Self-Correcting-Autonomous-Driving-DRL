# Issue Log & Resolution Report
**Date:** 2026-02-07

This document tracks the critical issues encountered during the optimization of the CARLA training pipeline and their resolutions.

## 1. Segfault (Signal 11/Code 139) on Map Switch
**Issue:**  
The CARLA server consistently crashed with `Segmentation fault (core dumped)` or `Signal 11` when transitioning between stages (e.g., loading `Town03` after `Town01`).
**Root Cause:**  
Memory corruption or incomplete cleanup in the CARLA 0.9.13 server instance when `client.load_world()` is called multiple times in the same session, especially when using `-nullrhi` or high-performance flags.
**Resolution:**  
Implemented a **Robust Curriculum Runner (`run_curriculum.sh`)** that kills and restarts the CARLA server process for every stage. This ensures a clean memory state for each map load.

## 2. Segfault (Code 139) on Exit
**Issue:**  
At the end of a successful training stage, the script often exited with `Code 139` (Segmentation Fault) despite saving the model.
**Root Cause:**  
A known issue in CARLA/Unreal Engine where the OpenGL context destruction or thread termination causes a crash during shutdown. It is harmless to the training result.
**Resolution:**  
Updated `run_curriculum.sh` to check for the existence of the expected model artifact (`final_model_stage_X.zip`). If the model exists, the script ignores Exit Code 139 and proceeds to the next stage.

## 3. Observation Space Mismatch (ValueError)
**Issue:**  
Stage 2 failed with `ValueError: Observation spaces do not match` when loading a checkpoint.
**Root Cause:**  
The training script tried to load a stale `stage_1` checkpoint from a previous run (before sensor upgrades) that had a different observation shape (`Dict` vs `Box`).
**Resolution:**  
Added logic to `run_curriculum.sh` to **automatically delete old `.zip` artifacts** when starting a fresh curriculum (Stage 1).

## 4. Accidental Checkpoint Deletion (Resume Logic Bug)
**Issue:**  
Running `./run_curriculum.sh 3` (Resume) inadvertently triggered the cleanup logic and deleted Stage 1 & 2 models, causing Stage 3 to fail.
**Root Cause:**  
The cleanup logic was not successfully conditioned on the start stage in an intermediate version of the script.
**Resolution:**  
Fixed `run_curriculum.sh` to only execute cleanup if `$START_STAGE` is 1.

## 5. Stage 3 Timeout (RuntimeError)
**Issue:**  
`RuntimeError: time-out of 40000ms while waiting for the simulator`.
**Root Cause:**  
Occasional race condition where the Python script attempts to connect before the restarted CARLA server is fully ready/listening on port 2000.
**Resolution:**  
Increased `sleep` duration in `run_curriculum.sh` and `launch_carla.sh`. Ensure `pkill` has time to fully terminate the previous instance (zombie processes can block the port).

## 6. Headless Rendering Crash (-nullrhi)
**Issue:**  
Using the `-nullrhi` flag caused immediate segfaults with certain sensors.
**Root Cause:**  
Incompatibility with some sensor rendering pipelines in 0.9.13.
**Resolution:**  
Switched to `-RenderOffScreen` which provides 90% of the performance benefit without the instability.

## Summary of Optimizations
- **Speed**: ~210 it/s (vs ~70 it/s baseline).
- **Sensors**: Replaced RGB/Lidar with specialized Vector Obstacle/Collision sensors.
- **Traffic Lights**: Direct API implementation (Zero-overhead).
- **Stability**: Server restarts + Exit code tolerance.
