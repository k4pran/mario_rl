from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent_base import Agent
from agent_q import AgentQ
from agent_random import AgentRandom

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_space = env.observation_space.shape
action_space = env.action_space.n


def play(agent: Agent, episodes=5000):

    done = False
    for step in range(episodes):

        print("Step: " + str(step))

        state = env.reset()
        while not done:
            agent.before()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            loss = agent.learn(state=state, action=action, reward=reward, next_state=next_state, done=done)

            agent.after()

            state = next_state

            # env.render()

            if loss:
                print("LOSS: {}".format(loss))

    env.close()

if __name__ == "__main__":
    play(AgentQ(state_space, action_space, batch_size=2))
