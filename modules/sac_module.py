import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from src.agent import Agent
from src.buffer import ReplayBuffer
from src.dataset import RLDataset
from src.reward import ProgressReward
from models.sac_model import SACPolicyNet, SACCriticNet
from omegaconf import DictConfig

class SACModule(pl.LightningModule):
    def __init__(self, env, config: DictConfig):
        super().__init__()
        self.automatic_optimization = False  # 自動最適化を無効化

        self.env = env
        obs_size = self.env.state_n
        action_size = self.env.action_n

        # configの内容をインスタンス変数に展開
        self.batch_size = config.batch_size
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic
        self.alpha = config.alpha
        self.tau = config.tau
        self.gamma = config.gamma
        self.replay_size = config.replay_size
        self.warm_start_steps = config.warm_start_steps
        self.episode_length = config.episode_length

        # アクターとクリティックのネットワークを初期化
        self.actor = SACPolicyNet(obs_size, action_size)
        self.critic = SACCriticNet(obs_size, action_size)  # クリティックネットワークを1つに統合
        self.target_critic = SACCriticNet(obs_size, action_size)  # ターゲットクリティックも1つに統合

        # ターゲットネットワークの初期化
        self.target_critic.load_state_dict(self.critic.state_dict())

        # リプレイバッファとエージェントの初期化
        self.buffer = ReplayBuffer(self.replay_size)
        self.reward = ProgressReward(map_manager=self.env.map_manager, reward_gain=0.4)
        self.agent = Agent(self.env, self.buffer, self.reward)
        self.populate(self.warm_start_steps)

    def populate(self, steps: int = None) -> None:
        """リプレイバッファを初期経験で満たします。"""
        steps = steps or self.warm_start_steps
        for _ in range(steps):
            self.agent.play_step(self.actor, self.device)

    def training_step(self, batch, batch_idx):
        # オプティマイザの取得
        actor_optimizer, critic_optimizer = self.optimizers()

        states, actions, rewards, dones, next_states = batch

        actions = actions.float()  
        states = states.float()
        next_states = next_states.float()
        dones = dones.float()


        # ターゲットQ値の計算
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)  # 統合されたクリティックネットワークから2つのQ値を取得
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # クリティックの損失計算
        current_q1, current_q2 = self.critic(states, actions)  # 同様に2つのQ値を取得
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # クリティックの最適化
        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        # アクターの損失計算
        actions_pred, log_probs = self.actor(states)
        q1, q2 = self.critic(states, actions_pred)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - min_q).mean()

        # アクターの最適化
        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        # ターゲットネットワークのソフト更新
        self.soft_update(self.critic, self.target_critic, self.tau)

        # ロギング
        self.log('critic_loss', critic_loss, prog_bar=True)
        self.log('actor_loss', actor_loss, prog_bar=True)

    def configure_optimizers(self):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        return [actor_optimizer, critic_optimizer]

    def soft_update(self, source_net, target_net, tau):
        """ターゲットネットワークをソフトに更新します。"""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def __dataloader(self):
        """リプレイバッファからデータローダーを初期化します。"""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    def train_dataloader(self):
        """トレインデータローダーを取得します。"""
        return self.__dataloader()

    @property
    def device(self):
        return next(self.actor.parameters()).device
