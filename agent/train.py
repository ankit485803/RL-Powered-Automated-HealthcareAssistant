

"""
Training Loop for RL Agent
Responsible: Ankit Kumar (Lead)
"""

from agent.rl_agent import RLAgent
from environment.healthcare_env import HealthcareEnvironment


def train_agent(episodes: int = 1000):
    """Main training loop"""
    
    env = HealthcareEnvironment()
    agent = RLAgent()
    
    for episode in range(episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(observation)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store experience and update agent
            # TODO: Ankit - Implement experience collection and PPO update
            
            observation = next_obs
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")
            agent.save(f"checkpoints/model_ep_{episode}.pt")
    
    print("Training complete")
    agent.save("checkpoints/final_model.pt")


if __name__ == "__main__":
    train_agent()