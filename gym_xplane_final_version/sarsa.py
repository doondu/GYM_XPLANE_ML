import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from collections import deque
import random
import os
import gym_xplane


# NUM_EPISODES = 10
# LEARNING_RATE_ACTOR = 0.0001
# LEARNING_RATE_CRITIC = 0.00046415888336127773
# BATCH_SIZE = 32
GAMMA = 0.98

# tensorflow 3: ERROR 수준 로그 메시지 출력
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


env = gym.make('gymXplane-v2')

# run_id = np.random.randint(10000)
#run_id = "no_render"
env.reset()

###########################################[YS] 일정한 데이터를 사용하기 위해 random seed 추가
env.observation_space.np_random.seed(123)
###########################################[YS]
# observation_space.sample()을 호출하여 환경에서 관측할 수 있는 상태의 예제를 10,000개 생성
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
print(observation_examples[1])
# Scikit-learn의 StandardScaler 객체를 생성하여 데이터를 표준화
scaler = sklearn.preprocessing.StandardScaler()
# StandardScaler 객체를 사용하여 observation_examples에 대한 평균과 표준편차를 계산
scaler.fit(observation_examples)

# FeatureUnion을 사용하여 여러 특징 추출기를 결합
featurizer = sklearn.pipeline.FeatureUnion([
        # RBFSampler 객체는 각기 다른 gamma 값을 사용하여 RBF (Radial Basis Function) 특징 맵을 생성
		("rbf1", RBFSampler(gamma=5.0, n_components=100)),
		("rbf2", RBFSampler(gamma=2.0, n_components=100)),
		("rbf3", RBFSampler(gamma=1.0, n_components=100)),
		("rbf4", RBFSampler(gamma=0.5, n_components=100))
])

# fit() 메서드는 모델을 학습시키고, transform() 메서드는 학습된 모델을 사용하여 데이터를 변환
# scaler는 전처리를 위한 학습 -> 변환환
# featurizer는 RBFSampler를 사용한 학습 -> 변환
featurizer.fit(scaler.transform(observation_examples))

#입력된 state를 scale과 featurized를 거치게 함
def featurize_state(state):
	try:
		scaled = scaler.transform([state])
	except:
		print("Except featurize_state,  stated: ",state)
	
	
	featurized = featurizer.transform(scaled)
	return featurized


#action space
class Policy():
	def __init__(self,lr,entropy_scalar):
		# Initialize policy graph
		with tf.variable_scope("policy"):
            # place holder? 실행할 때 외부에서 입력 데이터를 받아들이는 역할
			self.state_placeholder = tf.placeholder(tf.float32,shape=(None,400),name="state")
			self.target_placeholder = tf.placeholder(tf.float32,shape=(None),name = "target")
            # 가중치(weight) 업데이트 시에 현재의 기울기(gradient)에 곱해지는 스케일링 파라미터
			self.learning_rate = lr
            # 엔트로피 스칼라가 높을수록 정책은 더 많은 탐험을 하게 되며, 낮을수록 정책은 더 높은 보상을 추구하는 경향
			self.entropy_scalar = entropy_scalar


            # 평균
            # FCN layer, input: state, 1 output, activation function X, Xavier? 가중치를 적절히 초기화하여 학습을 안정화
			mu = tf.squeeze(tf.contrib.layers.fully_connected(
				inputs=self.state_placeholder,
				num_outputs=1,
				activation_fn=None,
				weights_initializer=tf.contrib.layers.xavier_initializer()
			))
            #표준 편차
			self.sigma = tf.squeeze(tf.contrib.layers.fully_connected(
				inputs=self.state_placeholder,
				num_outputs=1,
				activation_fn=None,
				weights_initializer=tf.contrib.layers.xavier_initializer()
			))
			self.sigma = tf.nn.sigmoid(self.sigma) 
            #정규 분포 객체를 생성

			dist = tf.distributions.Normal(loc=mu,scale=self.sigma)

            # 정책 신경망에서 얻은 정책의 확률 분포로부터 행동을 샘플링
            # dist에서 샘플을 추출하는 작업, 이 코드를 반복하여 실행하면 매번 다른 값이 출력
			act_1 =  tf.clip_by_value(dist.sample(1), env.action_space.low[0], env.action_space.high[0])
			act_2 =  tf.clip_by_value(dist.sample(1), env.action_space.low[1], env.action_space.high[1])
			act_3 =  tf.clip_by_value(dist.sample(1), env.action_space.low[2], env.action_space.high[2])
			act_4 =  tf.clip_by_value(dist.sample(1), env.action_space.low[3], env.action_space.high[3])
            #각 차원의 행동을 하나의 텐서
			self.action = tf.concat([act_1,act_2,act_3,act_4],0)


			# 로그 확률에 타겟 placeholder을 곱하여 손실을 계산, 보상 또는 어드밴티지 등과 같은 값
            # 위에서 계산한 두 항목을 모두 더한 후 음수로 바꾸어 손실 함수를 계산
			update = - (dist.log_prob(self.action) * self.target_placeholder) - (self.entropy_scalar * dist.entropy())

			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #Adam 옵티마이저를 사용하여 손실 함수를 최소화하는 방향으로 가중치를 업데이트
			self.updates = self.optimizer.minimize(update)


			"""
			policy_log = dist.log_prob(self.action)
			policy_log += self.entropy_scalar * dist.entropy()
			self.updates = []
			for v in tf.trainable_variables(scope="policy"):
				grad = tf.gradients(policy_log,v)[0]
				update = self.learning_rate * grad * self.target_placeholder
				self.updates.append(tf.assign_add(v, update, name='update'))
			"""
			
	def update(self,sess,state,target,act):
		state = featurize_state(state)
        # self.updates에 해당하는 연산을 실행 
        # 필요한 입력값은 feed_dict
		sess.run(self.updates,feed_dict={self.state_placeholder:state,self.target_placeholder:target,self.action:act})
		return 1
	
	def get_action(self,sess,state):
		state = featurize_state(state)
        # 계산된 action 반환
		return sess.run(self.action,feed_dict={self.state_placeholder:state})
	


class Critic():
	# Maps state to action
	def __init__(self,lr):
        # 400개의 특성(feature) line 42
		self.state_placeholder = tf.placeholder(tf.float32,shape=(None,400))
		self.target_placeholder = tf.placeholder(tf.float32,shape=(None))
		self.learning_rate = lr

        # state에 대한 value 예측
		self.out = tf.squeeze(tf.contrib.layers.fully_connected(
			inputs=self.state_placeholder,
			num_outputs=1,
			activation_fn=None,
			weights_initializer=tf.contrib.layers.xavier_initializer()
		))
		
		# We get the value of being in a state and taking an action

        # 입력된 두 텐서의 요소별 제곱 오차를 계산
		self.loss = tf.squared_difference(self.out,self.target_placeholder)

		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.step = optimizer.minimize(self.loss)
		
		
	def predict_value(self,sess,state):
		state = featurize_state(state)
		return sess.run(self.out,feed_dict={self.state_placeholder:state})
		
	def update(self,sess,state,target):
        # 최소화하는 방향으로 가중치를 업데이트
		state = featurize_state(state)		
		_,loss = sess.run([self.step,self.loss], feed_dict={self.state_placeholder:state,self.target_placeholder:target})
		return loss




def actor_critic(num_episodes,learning_rate_critic,learning_rate_actor,entropy_scalar):

	# random id for tf log
	run_id = np.random.randint(10000)

	tf.reset_default_graph()

	# Create our actor and critic
	actor = Policy(lr=learning_rate_actor,entropy_scalar=entropy_scalar)
	critic = Critic(lr=learning_rate_critic)

	#새로운 세션 생성
	sess = tf.Session()
	
    # TensorFlow 세션에서 모든 변수를 초기화하는 연산
	sess.run(tf.global_variables_initializer())
	filewriter = tf.summary.FileWriter(logdir="logs/" + str(run_id), graph=sess.graph)
	
	steps = 0

	scores = []
	checkFirst = True

	for e in range(num_episodes):
		
		state = env.reset()
		avg_critic_loss = 0
		avg_actor_loss = 0
		i=0
		score = 0
		
		while True:
			# Take a step
			#env.render()
			action = actor.get_action(sess,state)
			next_state,reward,done,_ = env.step(action)
			print('\n next state', next_state)
			if checkFirst == True:
				checkFirst = False
				print("PPPPPPPPPPPPAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSS")
				break
			# Append transition
			#memory.append([state,action,reward,next_state,done])

			#sample = random.sample(memory,1)

			s_state, s_action, s_reward, s_next_state, s_done = [state,action,reward,next_state,done]

            # SARSA logic, GAMMA is discount factor
			critic_target = s_reward + GAMMA * critic.predict_value(sess,s_next_state)
			td_error = critic_target - critic.predict_value(sess,s_state)

			critic_loss = critic.update(sess,s_state,critic_target)
			actor_loss = actor.update(sess,s_state,td_error,s_action)
				
			#action_value_summary = tf.Summary(value=[tf.Summary.Value(tag='Action Value',simple_value=action)])
			#filewriter.add_summary(action_value_summary,steps)
			#print('reward', s_reward)

			i += 1
			steps += 1
			#avg_actor_loss += 1
			#avg_critic_loss += critic_loss
			
			score += reward
			#print("Episode: " + str(e) + " Score: " + str(score))

			if done:
				break
			
			state = next_state
		print("Episode: " + str(e) + " Score: " + str(score))
		scores.append(score)
		#reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Reward',simple_value=score)])
		#filewriter.add_summary(reward_summary,e)

	return scores

s = actor_critic(num_episodes=100000000000,learning_rate_critic=1e-8,learning_rate_actor=1e-8,entropy_scalar=2.6366508987303556e-02)
# s = actor_critic(num_episodes=100000000000,learning_rate_critic=1,learning_rate_actor=1,entropy_scalar=2.6366508987303556e-02)

print(s)
