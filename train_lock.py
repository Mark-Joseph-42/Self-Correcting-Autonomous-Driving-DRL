#!/usr/bin/env python
"""
train_lock.py - Wrapper to ensure only one training process runs at a time.
"""
import os
import sys
import fcntl
import subprocess

LOCK_FILE = "/tmp/carla_train.lock"

def main():
    # Try to acquire lock
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("🔒 Lock acquired. Starting training...")
    except BlockingIOError:
        print("❌ Another training process is already running!")
        print("   Kill it with: pkill -9 -f train.py")
        sys.exit(1)
    
    # Run the actual training
    try:
        result = subprocess.run([sys.executable, "-u", "train.py"] + sys.argv[1:])
        sys.exit(result.returncode)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

if __name__ == "__main__":
    main()
