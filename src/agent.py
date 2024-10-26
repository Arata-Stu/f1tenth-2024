import numpy as np
import torch
from .buffer import Experience

class Agent:
    def __init__(self, env, replay_buffer, reward) -> None:
        """Agent class for SAC, handling interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.replay_buffer = replay_buffer

        self.max_steer = self.env.s_max
        self.max_speed = self.env.v_max

        self.obs = self.env.reset()
        self.state = convert_obs(self.obs)
        self.reward = reward

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.obs = self.env.reset()
        self.state = convert_obs(self.obs)

    @torch.no_grad()
    def get_action(self, actor_net, device) -> np.ndarray:
        """Samples an action from the actor network given the current state.

        Args:
            actor_net: Actor network
            device: current device

        Returns:
            action: action to take in the environment

        """

        # Convert state to tensor
        state = torch.tensor([self.state], dtype=torch.float32, device=device)
        # Sample action from the actor network
        action, _ = actor_net(state)
        # Detach and convert action to numpy array
        action = action.cpu().numpy()[0]
        # Rescale action if necessary (e.g., for environments with bounded actions)
        action = self.rescale_action(action)
        return action

    def rescale_action(self, action):
        """Rescales the action to the environment's action space if necessary.

        Args:
            action: action output from the actor network

        Returns:
            rescaled action

        """
        
        action = np.clip(action, -1.0, 1.0)
        return action

    @torch.no_grad()
    def play_step(self, actor_net, device):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            actor_net: Actor network
            device: current device

        Returns:
            reward: reward received after taking the action
            done: whether the episode has ended

        """
        actions = []
        nn_action = self.get_action(actor_net, device)
        action = convert_action(nn_action, max_steer=self.max_steer, max_speed=self.max_speed)
        # Interact with the environment
        actions.append(action)

        new_obs, reward, done, info = self.env.step(actions)

        reward = reward.calc_reward(pre_obs=self.obs, obs = new_obs)
        # Create an experience tuple
        new_state = convert_obs(new_obs)
        exp = Experience(self.state, action, reward, done, new_state)
        # Add experience to the replay buffer
        self.replay_buffer.append(exp)
        # Update the current state
        self.obs = new_obs
        self.state = new_state
        # Reset the environment if done
        if done:
            self.reset()
        return reward, done
    
def convert_action(nn_action, max_steer=0.4, max_speed=8.0):
        steering_angle = nn_action[0] * max_steer
        speed = (nn_action[1] + 1) * (max_speed  / 2 - 0.5) + 1
        speed = min(speed, max_speed) # cap the speed

        action = [steering_angle, speed]

        return action

def convert_obs(obs, id=0, max_scan_range=30.0):

    lidar_scans = np.array(obs['scans'][id])
    lidar_scans = np.clip(lidar_scans/max_scan_range, 0, 1)

    nn_obs = lidar_scans

    return nn_obs


