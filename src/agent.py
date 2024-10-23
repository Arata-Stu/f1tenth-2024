import numpy as np
import torch
from buffer import Experience

class Agent:
    def __init__(self, env, replay_buffer) -> None:
        """Agent class for SAC, handling interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = self.env.reset()

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
        action, _ = actor_net.sample(state)
        # Detach and convert action to numpy array
        action = action.cpu().numpy()[0]
        # Rescale action if necessary (e.g., for environments with bounded actions)
        # action = self.rescale_action(action)
        return action

    def rescale_action(self, action):
        """Rescales the action to the environment's action space if necessary.

        Args:
            action: action output from the actor network

        Returns:
            rescaled action

        """
        
        action = np.clip(action, 0, 10.0)
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
        action = self.get_action(actor_net, device)
        # Interact with the environment
        new_state, reward, done, info = self.env.step(action)
        # Create an experience tuple
        exp = Experience(self.state, action, reward, done, new_state)
        # Add experience to the replay buffer
        self.replay_buffer.append(exp)
        # Update the current state
        self.state = new_state
        # Reset the environment if done
        if done:
            self.reset()
        return reward, done
