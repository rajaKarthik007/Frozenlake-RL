import gym
from tensorboardX import SummaryWriter
import numpy as np
from collections import defaultdict

REWARD_THRESHOLD = 0.8
MAPSIZE = "4x4" # can be changed to 8x8 as well
SLIPPERY = True
ALPHA = 0.2
GAMMA = 0.9
UPDATE_STEPS = 10
TEST_EP_LIMIT = 20

class Agent():
    def __init__(self, env):
        self.env = env
        self.values = defaultdict(float)
        self.state, _ = self.env.reset()
        
    def best_action_and_value(self, state):
        qvals = [self.values[(state, action)] for action in range(self.env.action_space.n)]
        return max(qvals), np.argmax(qvals)
    
    def test_episodes(self, n, env):
        total_reward = 0.0
        for _ in range(n):
            self.state, _ = env.reset()
            while True:
                _, action = self.best_action_and_value(self.state)
                self.state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        return total_reward/n
    
    def sample_step_from_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = new_state
        return (old_state, action, new_state, reward)
    
    def q_learning_updates(self, n):
        for _ in range(n):
            s, a, snew, reward = self.sample_step_from_env()
            next_best_value, _ = self.best_action_and_value(snew)
            self.values[(s, a)] = (1 - ALPHA) * self.values[(s, a)] + ALPHA * ( reward + (GAMMA * next_best_value))
        
        
    
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, map_name=MAPSIZE)
    testenv = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, map_name=MAPSIZE)
    agent = Agent(env)
    best_reward = 0.0
    n_iter = 0
    writer = SummaryWriter(comment='-q-learning')
    while True:
        n_iter+=1
        agent.q_learning_updates(UPDATE_STEPS)
        avg_reward = agent.test_episodes(TEST_EP_LIMIT, testenv)
        writer.add_scalar("avg_reward", avg_reward, n_iter)
        if avg_reward > best_reward:
            print(f"updating best reward from {best_reward} to {avg_reward}")
            best_reward = avg_reward
        if avg_reward > REWARD_THRESHOLD:
            print(f"Solved in {n_iter} iterations")
            print(agent.values)
            break
        