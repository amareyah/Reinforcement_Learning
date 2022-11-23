import gym

env = gym.make("CartPole-v0")

env = gym.wrappers.Monitor(env,"Recordings",force=True)

for _ in range(5):

    env.reset()
    total_reward = 0
    total_steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("Total reward = {:.2f} and total steps = {:d}".format(total_reward,total_steps))
