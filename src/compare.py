import pandas as pd
import matplotlib.pyplot as plt
import os
from src.config import LOG_DIR

def compare_results():
    print("Generating comparison...")
    # Load curriculum logs
    curriculum_dfs = []
    for i in range(1, 4):
        f = os.path.join(LOG_DIR, f"stage_{i}.monitor.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, skiprows=1)
            df['stage'] = i
            curriculum_dfs.append(df)
            
    if not curriculum_dfs:
        print("No curriculum logs found.")
        return
        
    c_df = pd.concat(curriculum_dfs)
    c_df['cumulative_steps'] = c_df['l'].cumsum()
    
    # Load baseline logs
    b_f = os.path.join(LOG_DIR, "baseline.monitor.csv")
    if os.path.exists(b_f):
        b_df = pd.read_csv(b_f, skiprows=1)
        b_df['cumulative_steps'] = b_df['l'].cumsum()
    else:
        print("No baseline logs found.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(c_df['cumulative_steps'], c_df['r'].rolling(10).mean(), label='Curriculum')
    plt.plot(b_df['cumulative_steps'], b_df['r'].rolling(10).mean(), label='Baseline')
    plt.xlabel('Steps')
    plt.ylabel('Mean Reward')
    plt.title('Curriculum vs Baseline Training')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, "comparison.png"))
    
    # Summary Statistics
    summary = {
        'Curriculum Mean Reward (Final 10%)': c_df['r'].tail(len(c_df)//10).mean(),
        'Baseline Mean Reward (Final 10%)': b_df['r'].tail(len(b_df)//10).mean(),
        'Curriculum Total Steps': c_df['l'].sum(),
        'Baseline Total Steps': b_df['l'].sum()
    }
    pd.Series(summary).to_csv(os.path.join(LOG_DIR, "comparison_summary.csv"))
    print("Comparison complete. See results/ folder.")

if __name__ == "__main__":
    compare_results()
