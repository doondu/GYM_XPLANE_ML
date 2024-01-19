import argparse
import csv
import gym_xplane
#import p3xpc 
import gym
from datetime import datetime
from time import sleep, process_time, perf_counter, thread_time, time
import threading


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #client = p3xpc.XPlaneConnect()
    
    #parser.add_argument('--client', help='client address',default=client)
    parser.add_argument('--clientAddr', help='xplane host address', default='0.0.0.0')
    parser.add_argument('--xpHost', help='x plane port', default='127.0.0.1')
    parser.add_argument('--xpPort', help='client port', default=49009)
    parser.add_argument('--clientPort', help='client port', default=1)
    
    args = parser.parse_args()

    env = gym.make('gymXplane-v2')
    env.clientAddr = args.clientAddr
    env.xpHost = args.xpHost
    env.xpPort = args.xpPort
    env.clientPort = args.xpPort
    

    #env.seed(123)
    agent = RandomAgent(env.action_space)
    

    episodes = 0
    ##################################[YS]FILE
    # logName1 = "Log_step_" + str(datetime.now())
    # logName2 = "Log_epi_" + str(datetime.now())

    # logSt = 0
    # logEp = 0
    # rewardLast = 0
    # try:
    #     log_st = open(logName1, 'w', newline='')
    #     log_ep = open(logName2, 'w', newline='')
    #     logSt = csv.writer(log_st)
    #     logEp = csv.writer(log_ep)
    # except:
    #     print("################FAILED FILE OPEN################")
    ##################################[YS]FILE




    while episodes < 50:
        obs = env.reset()
        done = False
        ########################################[YS]LOGFILE, check episode's starting point
        # essStart = process_time()
        # erfStart = perf_counter()
        ########################################[YS]LOGFILE

        while not done:
            
            ########################################[YS]LOGFILE, check starting point
            # ssStart = process_time()
            # rfStart = perf_counter()
            #########################################[YS]LOGFILE

            #RL logic
            action = agent.act()
            obs, reward, done, _ = env.step(action) 
            print(obs, reward, done)

            ########################################[YS]LOGFILE, 
            # string1 = str(format((process_time() - ssStart), '.6f')) #processtime
            # string2 = str(format((perf_counter() - rfStart), '.6f')) #perfCounter
            # string3 = str(format(((reward) - rewardLast), '.6f')) #rewardVariance
            # string4 = str(threading.active_count())#threadCount

            # logSt.writerow([str(rfStart), string1, string2]) 
            #########################################[YS]LOGFILE
            rewardLast = reward
            print(f"########################{episodes}")

        episodes += 1
        ########################################[YS]LOGFILE
        # string5 = str(format((process_time() - essStart), '.6f'))#processtime
        # string6 = str(format((perf_counter() - erfStart), '.6f'))#perfcounter
        # logEp.writerow([str(erfStart), string5, string6])
        ########################################[YS]LOGFILE
    
    # logSt.close()
    # logEp.close()
    env.close()
