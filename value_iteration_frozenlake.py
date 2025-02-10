import gym
import numpy as np
from collections import defaultdict

from tensorboardX import SummaryWriter

GAMMA = 0.9
REWARD_THRESHOLD = 0.8

class Agent():
    def __init__(self, env):
        self.env = env
        self.rewards = defaultdict(float)
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.values = defaultdict(float)
    
    def play_n_random_steps(self, n):
        state, _ = self.env.reset()
        for _ in range(n):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.transitions[(state, action)][next_state] += 1
            self.rewards[(state, action, next_state)] = reward
            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state
    
    def action_value(self, state, action):
        total_visits = sum(self.transitions[(state, action)].values())
        action_value = 0.0
        for s, visits in self.transitions[(state, action)].items():
            action_value += ((visits/total_visits) * (self.rewards[(state, action, s)] + GAMMA*self.values[s]))
        return action_value
    
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            self.values[state] = max([self.action_value(state, action) for action in range(self.env.action_space.n)])
    
    def test_episodes(self, n):
        total_reward = 0.0
        for _ in range(n):
            state, _ = self.env.reset()
            while True:
                action = np.argmax([self.action_value(state, a) for a in range(self.env.action_space.n)])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                self.transitions[(state, action)][next_state] += 1
                self.rewards[(state, action, next_state)] = reward
                if terminated or truncated:
                    break
                else:
                    state = next_state
        return total_reward/n
    
    
if __name__ == "__main__":
    writer = SummaryWriter(comment = "-v-iter")
    agent = Agent(gym.make('FrozenLake-v1', is_slippery=True))
    n_iter = 0
    best_reward = 0.0
    while True:
        n_iter += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        avg_reward = agent.test_episodes(100)
        writer.add_scalar("reward", avg_reward, n_iter)
        if avg_reward > best_reward:
            print(f"updating best reward from {best_reward} to {avg_reward}")
            best_reward = avg_reward
        if avg_reward > REWARD_THRESHOLD:
            print(f"Solved in {n_iter} iterations")
            print(agent.values)
            break
        
        
    
    
    

