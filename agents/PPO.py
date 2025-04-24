from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.distributions import Normal


import gymnasium as gym


class PPO(nn.Module):
    """
    (Synchronous) Proximal Policy Optimization agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ), 
        ]

        self.actor_log_std = nn.Parameter(torch.zeros(n_actions))  # trainabile

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(list(self.actor.parameters()) + [self.actor_log_std], lr=actor_lr)

    def forward(self, x:np.ndaaray) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the action logits and value estimate for a given state."""
            
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)
        mean = self.actor(x)
        std = self.actor_log_std.exp()
        return (state_values, mean,std)
    
    def select_action(self, x: np.ndarray):
        """
        Selects an action based on the current state and the policy network.
        
        
        """
        state_values, mean, std = self.forward(x)
        dist = Normal(mean, std)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        actions = torch.tanh(actions)

        return actions, action_log_probs, state_values.squeeze(-1), entropy        
    
    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the losses for the actor and critic networks.
        
        Args:
            rewards: A tensor with the rewards, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state values, with shape [n_steps_per_update, n_envs].
            entropy: A tensor with the entropy of the actions, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks (1 if not done, 0 if done), with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE lambda parameter.
            ent_coef: The coefficient for the entropy term.
            device: The device to run the computations on.

        Returns:
            actor_loss: The loss for the actor network.
            critic_loss: The loss for the critic network.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device = device)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # compute advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        returns = advantages + value_preds
        
         # calculate the loss of the minibatch for actor and critic
        critic_loss = (
            (returns - value_preds).pow(2).mean()
        )

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)
    
    def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor) -> None:
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()