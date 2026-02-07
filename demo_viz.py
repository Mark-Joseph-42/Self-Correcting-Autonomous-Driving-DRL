import os
import time
import subprocess
import signal

def run_visual_demo():
    """
    Automated demo: 
    1. Launches CARLA Visualized.
    2. Runs Training for ~60s (approx 1000-2000 steps).
    3. Saves the model on interrupt.
    4. Runs Testing on that model.
    """
    print("🎬 Starting End-to-End Visual Demo (Training + Testing)...")
    os.environ["USE_CARLA"] = "1"
    
    # Clean old models to ensure we use NEW one
    print("🧹 Cleaning outputs...")
    subprocess.run("rm -rf outputs/stage_1/*", shell=True)
    subprocess.run("pkill -9 -f CarlaUE4 || true", shell=True)
    time.sleep(2)
    
    # 1. Start CARLA Viz
    print("🚀 Launching CARLA visualizer...")
    server_proc = subprocess.Popen(["./launch_carla_viz.sh"], shell=True, preexec_fn=os.setsid)
    time.sleep(25) # Wait for Town01 to load
    
    try:
        # 2. Training Burst
        print("\n🧠 STEP 1: Training Session (Live Rendering)")
        print("Starting PPO learning... (approx 60 seconds)")
        
        train_cmd = "conda run -n carla_py37 --no-capture-output python train.py"
        train_proc = subprocess.Popen(train_cmd, shell=True)
        
        # Wait 60s for training to perform several rollouts
        time.sleep(60)
        
        print("\n💾 Interrupting training and saving current model...")
        train_proc.send_signal(signal.SIGINT) # Trigger KeyboardInterrupt in train.py
        try:
            train_proc.wait(timeout=15)
        except:
            train_proc.kill()
            
        print("✅ Training burst complete.")
        
        # 3. Inference / Testing
        print("\n🚗 STEP 2: Testing Session (Inference)")
        print("Launching inference on the model we just trained...")
        
        # The model should be at ./outputs/stage_1/interrupted_model.zip
        model_path = "outputs/stage_1/interrupted_model.zip"
        if os.path.exists(model_path):
            env_vars = os.environ.copy()
            env_vars["TEST_MODEL_PATH"] = model_path
            subprocess.run("conda run -n carla_py37 python test.py", shell=True, env=env_vars)
        else:
            print("❌ Model not found! Searching stage_1 directory...")
            import glob
            files = glob.glob("outputs/stage_1/*.zip")
            if files:
                latest = max(files, key=os.path.getmtime)
                env_vars = os.environ.copy()
                env_vars["TEST_MODEL_PATH"] = latest
                subprocess.run("conda run -n carla_py37 python test.py", shell=True, env=env_vars)
            else:
                print("🛑 Critical Error: No model file generated.")

    finally:
        print("\n🛑 Shutting down simulator...")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except:
            pass
        print("✨ Demo finished.")

if __name__ == "__main__":
    run_visual_demo()
