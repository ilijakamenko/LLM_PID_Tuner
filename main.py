import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pid_controller import PIDController
from llm_pid_tuner import LLM_PID_Tuner
import time

plt.rcParams["font.family"] = "Times New Roman"

def main(pid_initial, plant, model_name, mode, aggressiveness):   
    closed_loop_initial = ctrl.feedback(pid_initial.tf()* plant)
    time_initial, response_initial = ctrl.step_response(closed_loop_initial)
    reference_signal = np.heaviside(time_initial, 1) 
    pid_tuned = pid_initial.copy()
    tuner=LLM_PID_Tuner(model_name=model_name, mode=mode, aggressiveness=aggressiveness) 
    plt.figure(figsize=(8, 3))
    results=[]
    for i in range(50):  # Stop early if criteria met
        closed_loop_tuned = ctrl.feedback(pid_tuned.tf() * plant)
        time_tuned, response_tuned = ctrl.step_response(closed_loop_tuned, time_initial)
        closed_loop_tuned = ctrl.feedback(pid_tuned.tf()* plant)
        time_tuned, response_tuned = ctrl.step_response(closed_loop_tuned, time_initial)
        plt.plot(time_tuned, response_tuned, 'k', label=f"iterations" if i == 1 else None, linestyle="dotted", linewidth=1)
        tuner.tune_pid(reference_signal, response_tuned, time_tuned, pid_tuned.Kp, pid_tuned.Ti, pid_tuned.Td)        
        print(f"Iteration {i}: Kp={pid_tuned.Kp:.3f}, Ti={pid_tuned.Ti:.3f}, Td={pid_tuned.Td:.3f}, tr: {tuner.metrics['rise_time']:.2f}s ({tuner.metrics['rise_time_%']:.2f}%), o: {tuner.metrics['overshoot']:.2f}% ({tuner.metrics['overshoot_%']:.2f}%), ts: {tuner.metrics['settling_time']:.2f}s ({tuner.metrics['settling_time_%']:.2f}%)")

        pid_tuned.update(tuner.Kp, tuner.Ti, tuner.Td)
        
        if tuner.metrics['early_stop']:  # Stop loop if early stopping triggered
            print("Early stop.")
            break
        time.sleep(1)
    plt.plot(time_tuned, response_tuned, 'r', label=f"last iteration",linestyle="solid", linewidth=2)
    plt.plot(time_initial, response_initial,'b', label="initial PID", linestyle="dashed", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Response")
    plt.legend()
    plt.grid(True)
    plt.show()   
    df = pd.DataFrame([{**entry['parameters'], **entry['metrics']} for entry in tuner.history])
    df = df.round(2)
    return df