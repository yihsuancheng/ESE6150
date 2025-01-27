import copy
# Constants that were previously obtained from the GPU device properties
max_threads_per_block = 1024  # Assuming typical max threads per block
max_block_dim_x = 1024  # Assuming typical max block dimension
max_square_block_dim = (int(max_block_dim_x**0.5), int(max_block_dim_x**0.5))
max_blocks = 65535  # Assuming typical max grid dimension on one axis
rec_max_control_rollouts = int(1e6)  # Theoretical limit, previously limited by max_blocks on GPU
rec_min_control_rollouts = 100
from PIL import Image
import numpy as np
filename = "/home/yihsuan/sim_ws/src/f1tenth_gym_ros/maps/levine_all.pgm"
img = Image.open(filename)

# Convert the image to a numpy array
img_array = np.array(img)
x_shape, y_shape = img_array.shape


class Config:
  
  """ Configurations that are typically fixed throughout execution. """
  
  def __init__(self, 
               T=10,  # Horizon (s)
               dt=0.1,  # Length of each step (s)
               num_control_rollouts=1024,  # Number of control sequences
               num_vis_state_rollouts=20,  # Number of visualization rollouts
               seed=1):
    
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T / dt)
    self.max_threads_per_block = max_threads_per_block  # saved as a constant

    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0
    
    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts
    if self.num_control_rollouts > rec_max_control_rollouts:
      self.num_control_rollouts = rec_max_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended max number of {}. (Max={})".format(
        rec_max_control_rollouts, max_blocks))
    elif self.num_control_rollouts < rec_min_control_rollouts:
      self.num_control_rollouts = rec_min_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended min number of {}. (Recommended max={})".format(
        rec_min_control_rollouts, rec_max_control_rollouts))
    
    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, self.num_control_rollouts])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])

import numpy as np
import time

DEFAULT_OBS_COST = 1e3
DEFAULT_DIST_WEIGHT = 10

# Stage costs (function)
def stage_cost(dist2, dist_weight):
  return dist_weight * dist2  # Squared term makes the robot move faster

# Terminal costs (function)
def term_cost(dist2, goal_reached):
  return (1 - float(goal_reached)) * dist2

class MPPI_Numba(object):
  
  """ 
  Implementation of Information theoretic MPPI by Williams et. al. without GPU support.
  Alg 2. in https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf
  """

  def __init__(self, cfg):
    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_control_rollouts = cfg.num_control_rollouts
    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.seed = cfg.seed

    # Initialize reusable variables
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    self.noise_samples = None
    self.u_cur = None
    self.u_prev = None
    self.costs = None
    self.weights = None
    self.state_rollout_batch = None  # For visualization only

    self.device_var_initialized = False
    self.reset()

  def reset(self):
    # Reset parameters
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    self.params = None
    self.params_set = False

    self.u_prev = None
    
    # Initialize all fixed-size variables ahead of time
    self.init_vars_before_solving()

  def init_vars_before_solving(self):
    if not self.device_var_initialized:
      t0 = time.time()
      
      self.noise_samples = np.zeros((self.num_control_rollouts, self.num_steps, 2), dtype=np.float32) # to be sampled collaboratively via CPU
      self.u_cur = np.copy(self.u_seq0)
      self.u_prev = np.copy(self.u_seq0)
      self.costs = np.zeros((self.num_control_rollouts), dtype=np.float32)
      self.weights = np.zeros((self.num_control_rollouts), dtype=np.float32)
      
      self.state_rollout_batch = np.zeros((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32)
      
      self.device_var_initialized = True
      print("MPPI planner has initialized memory after {} s".format(time.time()-t0))

  def setup(self, params):
        self.params = copy.deepcopy(params)
        self.params_set = True

  def check_solve_conditions(self):
        if not self.params_set:
            print("MPPI parameters are not set. Cannot solve")
            return False
        if not self.device_var_initialized:
            print("Device variables not initialized. Cannot solve.")
            return False
        return True

  def solve(self):
        if not self.check_solve_conditions():
            print("MPPI solve condition not met. Cannot solve. Return")
            return

        return self.solve_with_nominal_dynamics()
  
  def move_mppi_task_vars_to_device(self):
    # Since we're not using a GPU, we simply ensure all arrays are in the correct NumPy dtype
    vrange = self.params['vrange'].astype(np.float32)
    wrange = self.params['wrange'].astype(np.float32)
    xgoal = self.params['xgoal'].astype(np.float32)
    goal_tolerance = np.float32(self.params['goal_tolerance'])
    lambda_weight = np.float32(self.params['lambda_weight'])
    u_std = self.params['u_std'].astype(np.float32)
    x0 = self.params['x0'].astype(np.float32)
    dt = np.float32(self.params['dt'])

    if "obstacle_positions" in self.params:
        obs_pos = self.params['obstacle_positions'].astype(np.float32)
    else:
        # Dummy value as a placeholder when no obstacles are specified
        obs_pos = np.array([[1e5, 1e5]], dtype=np.float32)
    
    if "obstacle_radius" in self.params:
        obs_r = self.params['obstacle_radius'].astype(np.float32)
    else:
        # Dummy value as a placeholder
        obs_r = np.array([0], dtype=np.float32)

    obs_cost = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params else self.params['obs_penalty'])

    return vrange, wrange, xgoal, goal_tolerance, lambda_weight, u_std, x0, dt, obs_cost, obs_pos, obs_r

  def solve_with_nominal_dynamics(self):
    """
    Simulate nominal dynamics and adjust the cost function based on worst-case linear speed without using GPU kernels.
    """
    
    # Retrieve and set up all necessary parameters from self.params, assuming they have been converted to suitable formats
    vrange, wrange, xgoal, goal_tolerance, lambda_weight, \
           u_std, x0, dt, obs_cost, obs_pos, obs_r = self.move_mppi_task_vars_to_device()
  
    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    # Optimization loop
    for k in range(self.params['num_opt']):
        # Sample control noise
        self.noise_samples = self.sample_noise(self.num_control_rollouts, self.num_steps, u_std)

        # Rollout and compute mean or cvar
        self.costs = self.rollout_dynamics(vrange, wrange, xgoal, obs_cost, obs_pos, obs_r,
                                           goal_tolerance, lambda_weight, u_std, x0, dt, dist_weight,
                                           self.noise_samples, self.u_cur)

        # Remember the current control sequence as previous
        self.u_prev = np.copy(self.u_cur)

        # Compute cost and update the optimal control sequence
        self.update_control_sequence(self.costs, self.noise_samples, self.u_cur, lambda_weight, vrange, wrange)

    return self.u_cur
  
  def shift_and_update(self, new_x0, u_cur, num_shifts=1):
    self.params["x0"] = new_x0.copy()
    self.shift_optimal_control_sequence(u_cur, num_shifts)


  def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
    u_cur_shifted = u_cur.copy()
    u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
    self.u_cur_d = u_cur_shifted.astype(np.float32)

  def get_state_rollout(self):
    """
    Generate state sequences based on the current optimal control sequence.
    Assumes that relevant state and control parameters are stored in NumPy arrays.
    """

    assert self.params_set, "MPPI parameters are not set"

    # Check if the necessary variables have been initialized (now done on the CPU)
    if not self.device_var_initialized:
        print("Device variables not initialized. Cannot run mppi.")
        return
    
    # Prepare parameters using NumPy arrays on CPU instead of moving to GPU
    vrange = self.params['vrange'].astype(np.float32)
    wrange = self.params['wrange'].astype(np.float32)
    x0 = self.params['x0'].astype(np.float32)
    dt = np.float32(self.params['dt'])

    # Placeholder for future function that will perform state rollout on CPU
    self.state_rollout_batch = self.get_state_rollout_across_control_noise(
        x0, dt, self.noise_samples, vrange, wrange, self.u_prev, self.u_cur, self.num_vis_state_rollouts
    )
    
    return self.state_rollout_batch

  def rollout_dynamics(self, vrange, wrange, xgoal, obs_cost, obs_pos, obs_r,
                     goal_tolerance, lambda_weight, u_std, x0, dt, dist_weight, noise_samples, u_cur):
    num_rollouts, num_steps, _ = noise_samples.shape
    costs = np.zeros(num_rollouts)

    for bid in range(num_rollouts):
        x_curr = np.copy(x0)
        total_cost = 0
        for t in range(num_steps):
            noise_v, noise_w = noise_samples[bid, t]
            v_noisy = np.clip(u_cur[t, 0] + noise_v, vrange[0], vrange[1])
            w_noisy = np.clip(u_cur[t, 1] + noise_w, wrange[0], wrange[1])

            # Update state using simple kinematic model
            x_curr[0] += dt * v_noisy * np.cos(x_curr[2])
            x_curr[1] += dt * v_noisy * np.sin(x_curr[2])
            x_curr[2] += dt * w_noisy
            #print("x_cur", x_curr)
            dist_to_goal2 = np.sum((x_curr[:2] - xgoal)**2)
            total_cost += dist_weight * dist_to_goal2

            x_origin, y_origin = -3.47, -59.8
            x_indices, y_indices = (x_curr[0] - x_origin)/0.05, ((x_curr[1] - y_origin)/0.05 -x_shape)*-1
            #print(x_indices, y_indices)
            if img_array[int(y_indices), int(x_indices)] < 200:
               total_cost += obs_cost

            # for j in range(len(obs_pos)):
            #     dist_to_obs = np.sum((x_curr[:2] - obs_pos[j])**2) - obs_r[j]**2
            #     if dist_to_obs <= 0:
            #         total_cost += obs_cost

        goal_reached = dist_to_goal2 <= goal_tolerance**2
        total_cost += (1 - float(goal_reached)) * dist_to_goal2 * lambda_weight
        costs[bid] = total_cost

    return costs

  def update_control_sequence(self, costs, noise_samples, u_cur, lambda_weight, vrange, wrange):
    """
    Update the optimal control sequence based on previously evaluated cost values.
    This function uses NumPy operations to perform the updates previously handled by CUDA.
    """
    num_rollouts, num_steps, _ = noise_samples.shape

    # Compute weights from costs
    beta = np.min(costs)  # equivalent to the reduction operation in CUDA
    weights = np.exp(-1.0 / lambda_weight * (costs - beta))
    
    # Normalize weights
    weight_sum = np.sum(weights)
    normalized_weights = weights / weight_sum

    # Update the control sequence using weighted average of noise samples
    weighted_noise_sum = np.sum(noise_samples * normalized_weights[:, np.newaxis, np.newaxis], axis=0)
    u_cur += weighted_noise_sum
    #weighted_noise_sum = np.sum(noise_samples.transpose(1, 2, 0) * normalized_weights, axis=2)
    #u_cur += weighted_noise_sum.transpose()

    # Ensure control values stay within specified ranges
    for t in range(num_steps):
        u_cur[t, 0] = np.clip(u_cur[t, 0], vrange[0], vrange[1])
        u_cur[t, 1] = np.clip(u_cur[t, 1], wrange[0], wrange[1])

    return u_cur

  def get_state_rollout_across_control_noise(self, x0, dt, noise_samples, vrange, wrange, u_prev, u_cur, num_rollouts):
      """
      Simulate state sequences for visualization based on the current and previous control sequences.
      Assumes a fixed number of rollouts are performed. The first rollout uses the current best control sequence,
      and subsequent rollouts use random noise samples added to the previous control sequence for variability.
      """

      num_steps = u_cur.shape[0]
      state_rollout_batch = np.zeros((num_rollouts, num_steps + 1, 3))

      for bid in range(num_rollouts):
          # Initialize state from starting position
          x_curr = np.copy(x0)
          state_rollout_batch[bid, 0, :] = x_curr

          if bid == 0:
              # Use the current control sequence for the first rollout
              control_sequence = u_cur
          else:
              # Use noisy previous controls for other rollouts
              control_sequence = u_prev + noise_samples[bid]

          for t in range(num_steps):
              v = np.clip(control_sequence[t, 0], vrange[0], vrange[1])
              w = np.clip(control_sequence[t, 1], wrange[0], wrange[1])

              # Simulate dynamics
              x_next = np.array([
                  x_curr[0] + dt * v * np.cos(x_curr[2]),
                  x_curr[1] + dt * v * np.sin(x_curr[2]),
                  x_curr[2] + dt * w
              ])

              # Update current state and store in the rollout batch
              x_curr = x_next
              state_rollout_batch[bid, t + 1, :] = x_curr

      return state_rollout_batch



  def sample_noise(self, num_rollouts, num_steps, u_std):
      """
      Samples noise for control sequences. The function generates Gaussian noise 
      based on the provided standard deviations for each control input.

      Args:
      num_rollouts (int): The number of different control sequences for which noise is sampled.
      num_steps (int): The number of timesteps for each control sequence.
      u_std (np.array): An array of standard deviations for each control dimension, typically [std_v, std_w].

      Returns:
      np.array: A three-dimensional array of shape (num_rollouts, num_steps, 2) containing the sampled noise.
      """
      # Ensure u_std is an array for operations
      u_std = np.array(u_std, dtype=np.float32)
      
      # Sample noise for each control input dimension across all rollouts and steps
      noise_samples = np.random.normal(0, u_std, size=(num_rollouts, num_steps, 2))

      return noise_samples


