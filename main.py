from agent_q import AgentQ
from environment import make_env
from agent_frame import gym_runner

if __name__ == "__main__":
    env = make_env()
    action_space = env.action_space.n
    agent = AgentQ(action_space, 80, 80, training=True, load_agent=False)
    gym_runner.run(env, agent)
