from collections import deque
import pickle
import time
import numpy as np
import tensorflow as tf
import gym

#Exploration vs Exploitation
MAX_EPSILON = 1
MIN_EPSILON = 0.1
#Experience Replay
STARTING_SIZE = 50000
MINIBATCH_SIZE = 32
REPLAY_SIZE = 1000000
#Bellman's Equation
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0000625
#Neural Network
OBSERVATION_SIZE = 128
HIDDEN_SIZE = 400
ACTION_SIZE = 4
UPDATE_FREQUENCY = 10000
#Total number of Simulations
MAX_T = 9999999
MAX_FRAME = 10000000

#Paths
STARTING_EXPERIENCE_REPLAY_PATH = "experience_replay3.pickle"
EXPERIENCE_REPLAY_PATH = "experience_replay_progress.pickle"
PROGRESS_PATH = "progress.pickle"
PARAMETERS_PATH = "checkpoints_BRKOUT/breakout.ckpt"


class MainQN:
    '''
    Main NN for getting q-values based on observation which determines the action to be taken.
    Once target is calculated, pass in the required information into the tensorflow graph for
    the optimizer to adjust the weights accordingly.

    Functions:
    .get_Q() = get Q1 based on S1.
    .fit() = optimise and train the NN
    .get_weights() and .get_biases() = Get the value of NN parameters to be used in the target NN.
    '''
    def __init__(self, obs_size, hidden_size, action_size, learning_rate):
        self.hidden_size = hidden_size

        self.inputs_ = tf.placeholder(tf.float32, [None, obs_size], name="inputs")

        self.weights1 = tf.get_variable("w1", [obs_size, self.hidden_size], tf.float32,
                                        initializer=tf.initializers.truncated_normal(0, tf.sqrt(1/self.hidden_size)))
        self.weights2 = tf.get_variable("w2", [self.hidden_size, self.hidden_size], tf.float32,
                                        initializer=tf.initializers.truncated_normal(0, tf.sqrt(1/self.hidden_size)))
        self.weights3 = tf.get_variable("w3", [self.hidden_size, action_size], tf.float32,
                                        initializer=tf.initializers.truncated_normal(0, tf.sqrt(1/self.hidden_size)))
        self.bias1 = tf.get_variable("b1", [self.hidden_size], tf.float32,
                                     initializer=tf.zeros_initializer)
        self.bias2 = tf.get_variable("b2", [self.hidden_size], tf.float32,
                                     initializer=tf.zeros_initializer)
        self.bias3 = tf.get_variable("b3", [action_size], tf.float32,
                                     initializer=tf.zeros_initializer)

        self.fc1 = tf.nn.bias_add(tf.matmul(self.inputs_, self.weights1), self.bias1)
        self.fc1 = tf.nn.relu(self.fc1)

        self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, self.weights2), self.bias2)
        self.fc2 = tf.nn.relu(self.fc2)

        self.output = tf.nn.bias_add(tf.matmul(self.fc2, self.weights3), self.bias3)

        self.actions_ = tf.placeholder(tf.int64, [None])
        one_hot_actions = tf.one_hot(self.actions_, action_size)

        self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

        self.targetQ = tf.placeholder(tf.float32, [None], name="targetQ")

        #self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        #For deep networks, use this loss function instead to prevent gradient explosion.
        self.loss = tf.reduce_mean(tf.losses.huber_loss(self.targetQ, self.Q, 2, 0.5))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def get_Q(self, sess, obs):
        return sess.run(self.output, feed_dict={self.inputs_:obs})

    def fit(self, sess, obs, act, tar):
        feed_dict = {self.inputs_:obs, self.actions_:act, self.targetQ:tar}
        return sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
    #Convert tensors to np arrays for feed_dict into target graph
    def get_weights(self):
        return (np.array(self.weights1.eval()), np.array(self.weights2.eval()),
                np.array(self.weights3.eval()))

    def get_biases(self):
        return (np.array(self.bias1.eval()), np.array(self.bias2.eval()),
                np.array(self.bias3.eval()))

class TargetQN:
    '''
    The target network is used to get the q value given the new observation. The weights and biases
    are taken from the main network every so often. This seperation of networks allows a more stable
    target value and enable the loss function to converge easily.

    Functions:
    .get_Q() = get Q2 based on S2.
    '''
    def __init__(self, obs_size, hidden_size, action_size):
        self.hidden_size = hidden_size

        self.inputs_ = tf.placeholder(tf.float32, [None, obs_size], name="inputs")

        self.weights1 = tf.placeholder(tf.float32, [obs_size, self.hidden_size])
        self.weights2 = tf.placeholder(tf.float32, [self.hidden_size, self.hidden_size])
        self.weights3 = tf.placeholder(tf.float32, [self.hidden_size, action_size])
        self.bias1 = tf.placeholder(tf.float32, [self.hidden_size])
        self.bias2 = tf.placeholder(tf.float32, [self.hidden_size])
        self.bias3 = tf.placeholder(tf.float32, [action_size])

        self.fc1 = tf.nn.bias_add(tf.matmul(self.inputs_, self.weights1), self.bias1)
        self.fc1 = tf.nn.relu(self.fc1)

        self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, self.weights2), self.bias2)
        self.fc2 = tf.nn.relu(self.fc2)

        self.output = tf.nn.bias_add(tf.matmul(self.fc2, self.weights3), self.bias3)


    def get_Q(self, sess, obs, weights, biases):
        w1, w2, w3 = weights
        b1, b2, b3 = biases
        feed_dict = {self.weights1:w1, self.weights2:w2, self.weights3:w3,
                     self.bias1:b1, self.bias2:b2, self.bias3:b3, self.inputs_:obs}
        return sess.run(self.output, feed_dict=feed_dict)

def clip_epsilon(value, max_v, min_v):
    if value > max_v:
        value = max_v
    elif value < min_v:
        value = min_v
    return value

env = gym.make("Breakout-ram-v4")

class Memory:
    def __init__(self, replay_memory_size):
        self.experience = deque(maxlen=replay_memory_size)
    def add_memory(self, memory):
        self.experience.append(memory)
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.experience)),
                               size=batch_size,
                               replace=False)
        return [self.experience[i] for i in idx]

experience_replay = Memory(REPLAY_SIZE)

def gain_experience():
    #Start the environment and define observation
    env.reset()
    observation, reward, done, life = env.step(1)

    #Initiate Experience replay with experiences
    for _ in range(STARTING_SIZE):
        #env.render()
        action = env.action_space.sample()
        new_observation, reward, done, new_life = env.step(action)

        if done:
            new_observation = np.zeros(observation.shape)
            experience_replay.add_memory((observation, action, reward, new_observation))
            env.reset()
            observation, reward, done, life = env.step(1)
        elif life != new_life:
            new_observation = np.zeros(observation.shape)
            experience_replay.add_memory((observation, action, reward, new_observation))
            observation, reward, done, life = env.step(1)
        else:
            experience_replay.add_memory((observation, action, reward, new_observation))
            observation = new_observation

    print("Experience Replay Initiated! Saving in file...")
    pickle_out = open(STARTING_EXPERIENCE_REPLAY_PATH, "wb")
    pickle.dump(experience_replay, pickle_out)
    pickle_out.close()
    print(f"Experience Replay Saved at {STARTING_EXPERIENCE_REPLAY_PATH}")

def train(pre_trained, episode=1, frame=0, epsilon=1):
    if pre_trained:
        pickle_in = open(PROGRESS_PATH, "rb")
        progress = pickle.load(pickle_in)
        episode, frame, epsilon = progress
        pickle_in.close()
        pickle_in = open(EXPERIENCE_REPLAY_PATH, "rb")
        experience_replay = pickle.load(pickle_in)
        pickle_in.close()
        print(f"Pre Trained Experience Replay and Progress Loaded at {progress}")
    else:
        pickle_in = open(STARTING_EXPERIENCE_REPLAY_PATH, "rb")
        experience_replay = pickle.load(pickle_in)
        pickle_in.close()
        print("Starting Experience Replay Loaded!")

    main = MainQN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE, LEARNING_RATE)
    co_main = TargetQN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE)

    saver = tf.train.Saver()

    # Start the environment and define observation
    env.reset()
    observation, reward, done, life = env.step(1)
    with tf.Session() as sess:
        if pre_trained:
            saver.restore(sess, tf.train.latest_checkpoint("checkpoints_BRKOUT/", "checkpoint"))
            print("Model Restored!")
            temp_w = main.get_weights()
            temp_b = main.get_biases()
        else:
            sess.run(tf.global_variables_initializer())
        #Train the network
        loss_list = []
        reward_list = []
        start = time.time()
        # Frame is used instead of episodes because MAX_FRAME and frame count is used for updating
        # target q network
        while frame < MAX_FRAME:
            total_reward = 0
            total_loss = 0
            t = 0
            # Cant use for loop because when the simulation is done, loss still needs to be calculated
            while t < MAX_T:
                # Epsilon decay and clipping. This decides the probability of choosing a random action.
                if epsilon > MIN_EPSILON:
                    epsilon -= ((MAX_EPSILON - MIN_EPSILON) / 1000000)
                else:
                    epsilon = MIN_EPSILON

                # Choose action. Epsilon greedy policy for choosing an action.
                if epsilon > np.random.rand():
                    action = env.action_space.sample()
                else:
                    Q1 = main.get_Q(sess, observation.reshape((1, *observation.shape)))
                    action = np.argmax(Q1)
                new_observation, reward, done, new_life = env.step(action)
                total_reward += reward
                '''
                There are 3 scenarios in Breakout after a step. Each corresponding to different responses. 
                1. Life is lost Episode ends 
                2. Life is lost but episode doesn't end 
                3. No life is lost 
                '''
                if done:
                    new_observation = np.zeros(observation.shape)
                    experience_replay.add_memory((observation, action, reward, new_observation))
                    reward_list.append(total_reward) # total score in 1 episode
                    loss_list.append(total_loss/t) # loss per frame
                    if episode%50 == 0:
                        average_reward = sum(reward_list[-50:])/len(list(range(episode))[-50:])
                        average_loss = sum(loss_list[-50:])/len(list(range(episode))[-50:])
                        end = time.time()
                        print(f"Episode: {episode} Frame:{frame} Score:{average_reward} Epsilon:{epsilon} Loss:{average_loss} {end-start}")
                        start = end
                    env.reset()
                    observation, reward, done, life = env.step(1)
                    t = MAX_T

                elif life != new_life:
                    new_observation = np.zeros(observation.shape)
                    experience_replay.add_memory((observation, action, reward, new_observation))
                    observation, reward, done, life = env.step(1)
                    t += 1

                else:
                    experience_replay.add_memory((observation, action, reward, new_observation))
                    observation = new_observation
                    t += 1

                # Retrieve experiences in batches for optimization.
                batch = experience_replay.sample(MINIBATCH_SIZE)
                observations = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                new_observations = np.array([each[3] for each in batch])
                # Update target NN weights and biases
                if frame%UPDATE_FREQUENCY == 0:
                    temp_w = main.get_weights()
                    temp_b = main.get_biases()
                # Calculate Q2 with S2.
                Q2 = co_main.get_Q(sess, new_observations, temp_w, temp_b)
                # Set Q2 values to be zero when simulation ends and S2 is zero.
                episode_end = (new_observations == np.zeros(observations[0].shape)).all(axis=1)
                Q2[episode_end] = tuple([0]*ACTION_SIZE)
                # Calculate target value
                targets = rewards + DISCOUNT_FACTOR * np.max(Q2, axis=1)
                # Run Optimizer
                loss, _ = main.fit(sess, observations, actions, targets)

                total_loss += loss
                frame += 1

            #Saving network parameters, progress and experience replay
            if episode%500 == 0:
                saver.save(sess, PARAMETERS_PATH, global_step=episode)
                print(f"Network Parameters Saved at episode {episode}")

                pickle_out = open(PROGRESS_PATH, "wb")
                progress = (episode+1, frame, epsilon)
                pickle.dump(progress, pickle_out)
                pickle_out.close()
                print(f"Progress Saved at {progress}")

                pickle_out = open(EXPERIENCE_REPLAY_PATH, "wb")
                pickle.dump(experience_replay, pickle_out)
                pickle_out.close()
                print(f"Experience Replay with {len(experience_replay.experience)} experiences")

            episode += 1

#gain_experience()
train(True)

'''
Description of model:
Deep Q-learning Network with experience replay, seperate target Q network.

Improvements/Fine Tuning
-step(1) to fire each episode and each time life is lost. (Should work on reducing action space to 3)
-learning rate is reduced because Adam is used instead of RMSprop
-loss gradient clipping with huber_loss
-setting new_observation to zeros whenever life is lost. This is important
to let the network realise loosing a life is bad.
-frame skip is important to better compare with humans(who have a reaction time)
and also allow for faster training.
-because we used ram inputs, we do not have to deal with frame display errors and hence
inputs can just be a single frame.


'''
