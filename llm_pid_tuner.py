import control as ctrl
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

class LLM_PID_Tuner:
    def __init__(self, model_name="gpt-4o", mode="balanced", aggressiveness="moderate"):
        self.r = 0.0
        self.y = 0.0
        self.t = 0.0
        self.Kp = 0.0
        self.Ti = 0.0
        self.Td = 0.0
        self.history = []
        self.mode = mode
        self.aggressiveness = aggressiveness
        self.model_name = model_name
        self.conversation_history = []
        self.metrics = []
        self.goal=-50
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.ollama_host = os.getenv("OLLAMA_HOST")
        
        if self.model_name.startswith("gpt"):
            print(f"🔄 Using OpenAI API ({self.model_name})")
            self.client = OpenAI(api_key=self.api_key)
        else:
            print(f"🔄 Using Ollama API ({self.model_name})")
            self.client = OpenAI(base_url=self.ollama_host, api_key="ollama") 

    def compute_performance_metrics(self):
        """Computes performance metrics from step response using step_info."""
        info = ctrl.step_info(self.y, self.t)
    
        settling_time= info.get("SettlingTime", np.nan)
        rise_time=info.get("RiseTime", np.nan)
        overshoot=info.get("Overshoot", np.nan)
        
        if self.history!=[]:
            initial_metrics = self.history[0]['metrics']
            improvement_settling = ((settling_time - initial_metrics['settling_time']) / initial_metrics['settling_time']) * 100
            improvement_rise = ((rise_time - initial_metrics['rise_time']) / initial_metrics['rise_time']) * 100
            improvement_overshoot = ((overshoot - initial_metrics['overshoot']) / initial_metrics['overshoot']) * 100
        else:
            improvement_settling = 0
            improvement_rise = 0
            improvement_overshoot = 0
        
        self.metrics = {
            "early_stop": False,
            "rise_time": rise_time,
            "overshoot": overshoot,
            "settling_time": settling_time,
            "rise_time_%": improvement_rise,
            "overshoot_%": improvement_overshoot,
            "settling_time_%": improvement_settling
        }        
        
        self.history.append({"parameters": {"Kp": self.Kp, "Ti": self.Ti, "Td": self.Td}, "metrics": self.metrics})

        return self.metrics

    def extract_json(self, response_text):
        """Extract JSON block from AI response using regex."""
        json_pattern = r"\{.*?\}"
        match = re.search(json_pattern, response_text, re.DOTALL)
        return match.group(0) if match else None

    def call_llm_tuner(self):
        """Uses AI to suggest new PID parameters."""
        
        system_prompt = f"""
        You are an expert in control engineering specializing in PID tuning. Your goal is to iteratively refine the PID controller gains (Kp, Ti, Td) for a given system to optimize performance.

        The user provides system response metrics (settling time, rise time, overshoot) and the current PID gains.
        Based on these values, suggest improved PID parameters to meet the specified tuning mode:

        - 'speedup': Minimize rise and settling time while ensuring stability,
        - 'reduce_overshoot': Reduce overshoot as much as possible while maintaining acceptable response times,
        - 'balanced': Improve all three metrics together, ensuring a trade-off between response speed and stability.

        The user also specifies the aggressiveness level:
        - 'aggressive': Make larger changes (40-50% per step),
        - 'moderate': Make medium changes (10-30% per step),
        - 'fine': Make small changes (1-5% per step).

        Ensure the new PID gains do not destabilize the system.
        Respond in JSON format: {{"Kp": value, "Ti": value, "Td": value}}
        """
        #After JSON add explanation.

       
        
        #Give only values in JSON structure without any additions.
        #Everytime try to make updatas of parameters.
        user_message = f"""
        The system's current performance:
        - Rise time: {self.metrics["rise_time"]:.2f} sec
        - Overshoot: {self.metrics["overshoot"]:.2f} %
        - Settling time: {self.metrics["settling_time"]:.2f} sec

        The current PID gains are:
        - Kp = {self.Kp}
        - Ti = {self.Ti}
        - Td = {self.Td}

        The tuning mode is '{self.mode}':

        The tuning aggressiveness is '{self.aggressiveness}'.
        Suggest new values for Kp, Ti, and Td to improve performance.
        """

        if not any(msg['role'] == 'system' for msg in self.conversation_history):
            self.conversation_history.append({"role": "system", "content": system_prompt})
        self.conversation_history.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                max_tokens=200,
                temperature=0
            )
            gpt_output = response.choices[0].message.content.strip()
            #print(gpt_output)
            self.conversation_history.append({"role": "assistant", "content": gpt_output})

            extracted_json = self.extract_json(gpt_output)
            if extracted_json:
                self.Kp=json.loads(extracted_json)["Kp"]
                self.Ti=json.loads(extracted_json)["Ti"]
                self.Td=json.loads(extracted_json)["Td"]  
            else:
                print("No valid JSON found in AI response, using fallback values.")

        except Exception as e:
            print(f"Error querying AI: {e}, using fallback values.")

    def tune_pid(self,r, y, t, Kp, Ti, Td):
        """Refines PID parameters iteratively while ensuring system stability and applying early stopping criteria."""
        self.r = r
        self.y = y
        self.t = t
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        
        self.compute_performance_metrics()

        # Early stopping criteria
        if self.mode == 'speedup' and self.history[-1]['metrics']['rise_time_%'] <= self.goal:
            self.history[-1]['metrics']["early_stop"]=True
        if self.mode == 'reduce_overshoot' and self.history[-1]['metrics']['overshoot_%'] <= self.goal:
            self.history[-1]['metrics']["early_stop"]=True
        if self.mode == 'balanced' and (self.history[-1]['metrics']['overshoot_%'] <=  self.goal or self.history[-1]['metrics']['rise_time_%'] <= self.goal):
            self.history[-1]['metrics']["early_stop"]=True
            
        if self.history[-1]['metrics']["early_stop"]==False:
            self.call_llm_tuner()
            