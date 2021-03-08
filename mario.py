from time import sleep

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from agent_base import Agent
from agent_q import AgentQ, make_env
from agent_random import AgentRandom
from image_viewer import display_img

env = make_env('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)
state_space = env.observation_space.shape
action_space = env.action_space.n

training = True
render = False

video_recorder = VideoRecorder(env, path="./output/test.mp4")
video_recorder.enabled = True

def write_summary(episode, total_score, steps, distance):
    print("Writing summary")
    f = open("summaries.txt", "a")
    f.write("episode: {} | score: {} | steps: {} | distance: {}\n".format(episode, total_score, steps, distance))
    f.close()


def play(agent: Agent, episodes=1000000):

    for episode in range(episodes):

        print("Episode: " + str(episode))

        state = env.reset()
        done = False

        total_score = 0
        steps = 0
        distance = 0
        while not done:
            agent.before()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            print("Game score: {} time: {} ({}, {})".format(info['score'], info['time'], info['x_pos'], info['y_pos']))
            if info['x_pos'] > distance:
                distance = info['x_pos']

            loss = agent.learn(state=state, action=action, reward=reward, next_state=next_state, done=done)

            agent.after(total_score)

            state = next_state

            if render:
                sleep(0.08)
                env.render()
                for i in range(6):
                    video_recorder.capture_frame()

            total_score += reward
            if training:
                if loss:
                    print("SCORE: {} REWARD: {} LOSS: {}".format(total_score, reward, loss))

            steps += 1

        write_summary(episode, total_score, steps, distance)

    env.close()


if __name__ == "__main__":
    play(AgentQ(action_space, 80, 80, training=training, load_agent=False))
    video_recorder.close()
    video_recorder.enabled = False
    env.close()
