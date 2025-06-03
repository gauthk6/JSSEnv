# JSSEnv/envs/jss_env.py
import bisect
import datetime
import random

import pandas as pd
import gymnasium as gym # Use Gymnasium
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path


class JssEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_config=None):
        super().__init__() 

        if env_config is None:
            instance_file_name = "ta01" 
            print(f"WARNING: env_config is None. Using default instance: {instance_file_name} for JssEnv initialization.")
            base_path = Path(__file__).parent.absolute()
            instance_path_default = base_path / "instances" / instance_file_name
            env_config = {"instance_path": str(instance_path_default)}
        
        instance_path_str = env_config.get("instance_path", None)
        if instance_path_str is None:
            raise ValueError("instance_path must be provided in env_config")
        
        instance_path = Path(instance_path_str)
        if not instance_path.exists():
            raise FileNotFoundError(f"JSSP instance file not found at: {instance_path}")

        # --- Instance Data Loading ---
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None 
        self.jobs_length = None 
        self.max_time_op = 0 
        self.max_time_jobs = 0 
        self.sum_op = 0 
        
        with open(instance_path, "r") as instance_file:
            for line_cnt, line_str in enumerate(instance_file, start=1):
                split_data = list(map(int, line_str.split()))
                if line_cnt == 1:
                    self.jobs, self.machines = split_data
                    self.instance_matrix = np.zeros((self.jobs, self.machines), dtype=[('machine', 'i4'), ('time', 'i4')])
                    self.jobs_length = np.zeros(self.jobs, dtype=int)
                else:
                    job_nb = line_cnt - 2
                    op_idx = 0
                    for i in range(0, len(split_data), 2):
                        machine, time = split_data[i], split_data[i + 1]
                        if job_nb < self.jobs and op_idx < self.machines:
                            self.instance_matrix[job_nb, op_idx] = (machine, time)
                            self.max_time_op = max(self.max_time_op, time)
                            self.jobs_length[job_nb] += time
                            self.sum_op += time
                        op_idx += 1
        
        if self.max_time_op == 0 and self.jobs > 0 :
            print("Warning: Max operation time is 0. Instance might be empty or invalid.")
            self.max_time_op = 1 
        self.max_time_jobs = np.max(self.jobs_length) if self.jobs > 0 and len(self.jobs_length) > 0 else 0
        
        assert self.jobs > 0, "Number of jobs must be greater than 0."
        assert self.machines > 0, "Number of machines must be greater than 0."
        if self.jobs > 0 :
             assert self.max_time_op > 0, "Max operation time must be positive."
             assert self.max_time_jobs > 0, "Max job length must be positive."
        assert self.instance_matrix is not None

        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        self.NO_OP_ACTION = self.jobs

        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.jobs + 1,), dtype=bool),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs, 7), dtype=np.float32)
        })
        
        self.colors = [tuple([random.random() for _ in range(3)]) for _ in range(self.machines)]
        self.start_timestamp_render = 0

        # --- Manual Truncation Step Counter ---
        self.current_episode_steps = 0
        # Fixed max_episode_steps that works for all problem sizes
        self.max_episode_steps = 3000  # Plenty of steps for any JSSP instance

        # --- State Variables (will be initialized in reset) ---
        self.solution = None
        self.current_time_step = 0.0
        self.next_time_step_events = [] 
        self.legal_actions = np.zeros(self.jobs + 1, dtype=bool)
        self.time_until_available_machine = np.zeros(self.machines, dtype=float)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=float)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int) 
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=float)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int) 
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=float)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=float)
        self.state_obs_buffer = np.zeros((self.jobs, 7), dtype=np.float32)
        self.nb_legal_actions = 0 
        self.nb_machine_legal = 0 
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        self._prioritization_rules_active = True # Control if heuristics run (kept for consistency)

    def _update_state_obs_buffer(self):
        for j in range(self.jobs):
            self.state_obs_buffer[j, 0] = self.legal_actions[j]
            self.state_obs_buffer[j, 1] = self.time_until_finish_current_op_jobs[j] / self.max_time_op if self.max_time_op > 0 else 0
            self.state_obs_buffer[j, 2] = self.todo_time_step_job[j] / self.machines if self.machines > 0 else 0
            self.state_obs_buffer[j, 3] = self.total_perform_op_time_jobs[j] / self.max_time_jobs if self.max_time_jobs > 0 else 0
            current_op_idx = self.todo_time_step_job[j]
            if current_op_idx < self.machines:
                machine_needed = self.instance_matrix[j, current_op_idx]['machine']
                wait_time_for_machine = self.time_until_available_machine[machine_needed]
                self.state_obs_buffer[j, 4] = wait_time_for_machine / self.max_time_op if self.max_time_op > 0 else 0
            else: 
                self.state_obs_buffer[j, 4] = 0.0 
            self.state_obs_buffer[j, 5] = self.idle_time_jobs_last_op[j] / self.sum_op if self.sum_op > 0 else 0
            self.state_obs_buffer[j, 6] = self.total_idle_time_jobs[j] / self.sum_op if self.sum_op > 0 else 0
        return {"real_obs": self.state_obs_buffer.copy(), "action_mask": self.legal_actions.copy()}

    def _recalculate_legal_actions_and_machine_status(self):
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        self.machine_legal.fill(False)
        for job_idx in range(self.jobs):
            if self.todo_time_step_job[job_idx] < self.machines: 
                current_op_idx = self.todo_time_step_job[job_idx]
                machine_needed = self.instance_matrix[job_idx, current_op_idx]['machine']
                if self.time_until_available_machine[machine_needed] == 0:
                    self.legal_actions[job_idx] = True
                    self.nb_legal_actions += 1
                    if not self.machine_legal[machine_needed]:
                        self.machine_legal[machine_needed] = True
                        self.nb_machine_legal += 1
                else:
                    self.legal_actions[job_idx] = False
            else: 
                self.legal_actions[job_idx] = False
        if self._prioritization_rules_active:
            self._apply_prioritization_rules()
        self._check_no_op_legality()

    def _apply_prioritization_rules(self):
        pass # Placeholder for your prioritization heuristics if any

    def _check_no_op_legality(self):
        if self.nb_legal_actions == 0:
            self.legal_actions[self.NO_OP_ACTION] = bool(len(self.next_time_step_events) > 0)
        else: 
            self.legal_actions[self.NO_OP_ACTION] = True # Generally allow NO-OP if jobs are also legal (agent can choose to wait)
        
        if np.sum(self.legal_actions) == 0 and len(self.next_time_step_events) > 0:
            self.legal_actions[self.NO_OP_ACTION] = True


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_episode_steps = 0 # Reset manual step counter
        self.current_time_step = 0.0
        self.next_time_step_events = [] 
        self.time_until_available_machine.fill(0.0)
        self.time_until_finish_current_op_jobs.fill(0.0)
        self.todo_time_step_job.fill(0)
        self.total_perform_op_time_jobs.fill(0.0)
        self.total_idle_time_jobs.fill(0.0)
        self.idle_time_jobs_last_op.fill(0.0)
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        for job_idx in range(self.jobs):
            self.needed_machine_jobs[job_idx] = self.instance_matrix[job_idx, 0]['machine']
        self._recalculate_legal_actions_and_machine_status()
        return self._update_state_obs_buffer(), {}

    def _increase_time_to_next_event(self):
        if not self.next_time_step_events: return 0.0
        self.next_time_step_events.sort() 
        next_event_time, _ = self.next_time_step_events.pop(0) # Job_idx also popped if stored
        time_advanced = next_event_time - self.current_time_step
        if time_advanced < 0: time_advanced = 0
        self.current_time_step = next_event_time
        for j in range(self.jobs):
            if self.time_until_finish_current_op_jobs[j] > 0:
                processed_during_advance = min(self.time_until_finish_current_op_jobs[j], time_advanced)
                self.time_until_finish_current_op_jobs[j] -= processed_during_advance
                self.total_perform_op_time_jobs[j] += processed_during_advance
                if self.time_until_finish_current_op_jobs[j] == 0:
                    self.todo_time_step_job[j] += 1
                    if self.todo_time_step_job[j] < self.machines:
                        self.needed_machine_jobs[j] = self.instance_matrix[j, self.todo_time_step_job[j]]['machine']
                        self.idle_time_jobs_last_op[j] = 0 
                    else: 
                        self.needed_machine_jobs[j] = -1 
            elif self.todo_time_step_job[j] < self.machines :
                 self.idle_time_jobs_last_op[j] += time_advanced
                 self.total_idle_time_jobs[j] += time_advanced
        for m in range(self.machines):
            if self.time_until_available_machine[m] > 0:
                self.time_until_available_machine[m] = max(0, self.time_until_available_machine[m] - time_advanced)
        self._recalculate_legal_actions_and_machine_status()
        return time_advanced

    def _are_all_jobs_completed(self):
        return all(self.todo_time_step_job[j] >= self.machines for j in range(self.jobs))

    def _is_done(self): # This determines 'terminated_from_env_rules'
        all_completed = self._are_all_jobs_completed()
        if all_completed:
            # print(f"DEBUG JSSEnv: _is_done() -> True [Natural Completion] at time {self.current_time_step:.2f}")
            return True
        # Check for deadlock only if not all jobs are completed
        current_legal_actions_sum = np.sum(self.legal_actions)
        if current_legal_actions_sum == 0:
            # print(f"DEBUG JSSEnv: _is_done() -> True [Deadlock - No Legal Actions] at time {self.current_time_step:.2f}. Mask: {self.legal_actions.astype(int)}. Future events: {len(self.next_time_step_events)}")
            return True
        return False

    def _reward_scaler(self, reward_value: float) -> float:
        if self.max_time_op == 0: return reward_value
        return reward_value / self.max_time_op

    def step(self, action: int):
        self.current_episode_steps += 1 # Increment manual step counter

        # --- CRITICAL WARNING check from before (good to keep) ---
        current_mask_at_step_entry = self.legal_actions.astype(bool).copy()
        if not current_mask_at_step_entry[action]:
            print(f"JSSEnv CRITICAL WARNING: Agent selected action {action}, but current_mask_at_step_entry[{action}] is FALSE!")
            print(f"Full mask at step entry was: {current_mask_at_step_entry.astype(int)}")
        # ---

        time_advanced_this_step = 0.0
        info = {"makespan": None, "all_jobs_completed": False} # Initialize info

        if action == self.NO_OP_ACTION:
            if len(self.next_time_step_events) > 0:
                time_advanced_this_step = self._increase_time_to_next_event()
        else: 
            job_to_schedule = action
            current_op_idx = self.todo_time_step_job[job_to_schedule]
            machine_needed = self.instance_matrix[job_to_schedule, current_op_idx]['machine']
            
            # This assertion might still be useful during debugging agent behavior
            # but should ideally not fail if agent respects masks.
            assert self.time_until_available_machine[machine_needed] == 0, \
                f"Machine {machine_needed} not free for job {job_to_schedule} (op {current_op_idx}) at time {self.current_time_step}. Mask was {current_mask_at_step_entry.astype(int)}"

            op_time = self.instance_matrix[job_to_schedule, current_op_idx]['time']
            self.time_until_available_machine[machine_needed] = op_time
            self.time_until_finish_current_op_jobs[job_to_schedule] = op_time
            self.solution[job_to_schedule, current_op_idx] = self.current_time_step
            completion_time = self.current_time_step + op_time
            # Add event: (time, job_idx). Store job_idx for _increase_time_to_next_event if it needs it.
            # For simplicity, if _increase_time_to_next_event only pops time, job_idx isn't strictly needed in tuple.
            bisect.insort(self.next_time_step_events, (completion_time, job_to_schedule))
            self.idle_time_jobs_last_op[job_to_schedule] = 0
            self._recalculate_legal_actions_and_machine_status()
            if self.nb_legal_actions == 0 and len(self.next_time_step_events) > 0:
                time_advanced_this_step = self._increase_time_to_next_event()
        
        step_reward_raw = -time_advanced_this_step if time_advanced_this_step > 0 else -0.01 

        terminated_from_env_rules = self._is_done()
        truncated_by_step_limit = False
        if self.current_episode_steps >= self.max_episode_steps:
            truncated_by_step_limit = True
            # print(f"DEBUG JSSEnv: MANUAL TRUNCATION at episode step {self.current_episode_steps}, global time {self.current_time_step:.2f}")

        if terminated_from_env_rules:
            all_jobs_completed = self._are_all_jobs_completed()
            info['all_jobs_completed'] = all_jobs_completed
            info['makespan'] = self.current_time_step
            # print(f"DEBUG JSSEnv STEP: Terminated by JSS rules. All jobs completed: {all_jobs_completed}. Makespan: {self.current_time_step:.2f}")
            if not all_jobs_completed:
                scaled_reward = -10.0 # Absolute penalty, not scaled by _reward_scaler
                if self.current_time_step == 0 and action == self.NO_OP_ACTION : 
                     scaled_reward = -50.0 # Harsher absolute penalty
            else: 
                scaled_reward = self._reward_scaler(step_reward_raw) + 1.0 # Base step reward + completion bonus
        elif truncated_by_step_limit:
            info['all_jobs_completed'] = self._are_all_jobs_completed() # Check status at truncation
            info['makespan'] = self.current_time_step # Current makespan at truncation
            # print(f"DEBUG JSSEnv STEP: Truncated by step limit. Makespan: {self.current_time_step:.2f}")
            scaled_reward = self._reward_scaler(step_reward_raw) 
        else: 
            scaled_reward = self._reward_scaler(step_reward_raw)
            # No need to set makespan/all_jobs_completed in info if not done
        
        return self._update_state_obs_buffer(), scaled_reward, terminated_from_env_rules, truncated_by_step_limit, info

    def render(self, mode="human"): # Keep your existing render or use this example
        if mode == "human":
            df = []
            for job_idx in range(self.jobs):
                for op_idx in range(self.machines):
                    start_time = self.solution[job_idx, op_idx]
                    if start_time != -1: 
                        op_duration = self.instance_matrix[job_idx, op_idx]['time']
                        machine_used = self.instance_matrix[job_idx, op_idx]['machine']
                        df.append(dict(Task=f"Job {job_idx}", Start=start_time, Finish=start_time + op_duration, Resource=f"Machine {machine_used}", Job=job_idx))
            if not df: return None
            df = pd.DataFrame(df)
            job_colors = {job_idx: f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})' for job_idx in range(self.jobs)}
            fig = ff.create_gantt(df, index_col='Resource', colors=job_colors, show_colorbar=True, group_tasks=True, title=f"Job Shop Schedule (Time: {self.current_time_step:.2f})")
            fig.update_yaxes(autorange="reversed") 
            fig.show()
            return fig
        return None

    def close(self):
        pass