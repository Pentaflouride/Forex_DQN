from environment import ForexEnv
from DQN_agent import ForexAgent
import numpy as np

# Initialize environment and agent
env = ForexEnv(symbol="EURUSD", episode_length=1440)
state_size = len(env.reset())
action_size = 4  # hold, buy, sell, wait
agent = ForexAgent(state_size, action_size)

# Training loop
n_episodes = 3
batch_size = 32

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    state = state.astype('float32')
    total_reward = 0

    for time in range(1440):  # 1440 minutes in a day
        action = agent.act(state, env.position)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        print(f"Episode: {e}/{n_episodes}, Action: {action}, e: {agent.epsilon:.2}, Total reward: {reward} ")
        if done:
            print(f"Episode: {e}/{n_episodes}, Score: {time}, e: {agent.epsilon:.2}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if e % 10 == 0:
        agent.update_target_model()

# Save the trained model
agent.save("forex_dqn.h5")
