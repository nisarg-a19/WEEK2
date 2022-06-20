import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time


class Graph:
    def __init__(self, num_of_nodes, directed=True):
        self.m_num_of_nodes = num_of_nodes
        self.m_directed = directed
        self.m_nodes = range(self.m_num_of_nodes)
        # Initialize the adjacency matrix
        # Create a matrix with `num_of_nodes` rows and columns
        self.m_adj_matrix = [[0 for column in range(num_of_nodes)] 
                            for row in range(num_of_nodes)]
        self.m_adj_list = {node: list() for node in self.m_nodes}  
    def add_edge(self, node1, node2, weight=1):
        self.m_adj_matrix[node1][node2] = weight

        if not self.m_directed:
            self.m_adj_matrix[node2][node1] = weight
        
        self.m_adj_list[node1].append((node2, weight))
        
        if not self.m_directed:
        	self.m_adj_list[node2].append((node1, weight))
    
    def print_adj_matrix(self):
        print(self.m_adj_matrix)
    
    def print_adj_list(self):
        for key in self.m_adj_list.keys():
            print("node", key, ": ", self.m_adj_list[key])
    def adj_matrix(self):
      return self.m_adj_matrix
    def adj_list(self):
      return self.m_adj_list

graph = Graph(30)

for i in range(60):
  node1 = np.random.randint(0,30)
  node2 = np.random.randint(0,30)
  weight = np.random.randint(0,100)
  graph.add_edge(node1,node2,weight)
mat = graph.adj_matrix()
lst = graph.adj_list()



style.use("ggplot")

SIZE = 30

HM_EPISODES = 25000

PENALTY = 300
REWARD = 25
epsilon = 0.0
EPS_DECAY = 0.9998
SHOW_EVERY = 3000  

start_q_table ="qtable.pickle"

LEARNING_RATE = 0.1
DISCOUNT = 0.95
START_N = 1
END_N = 2 



d = {1: (255, 175, 0),
     2: (0, 255, 0)
     }


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        

    def __str__(self):
        return f"{self.x}"

    def __sub__(self, other):
        return (self.x-other.x)

    def action(self, choice,lst):
        '''
        
        '''
        if(len(lst[self.x]) ==0):
          self.x = np.random.randint(0,SIZE)
        elif choice <len(lst[self.x]):
          flag = False;
          for i in range(len(lst[self.x])):
            if choice==i:
              flag = True
              self.x = lst[self.x][i][0]
          if flag==False:
            self.x = lst[self.x][0][0]
        else:
          self.x = lst[self.x][0][0]


if start_q_table is None:

    q_table = {}
    for i in range(-SIZE+1, 1):

          q_table[i] = [np.random.randint(-SIZE+1, 1) for i in range(SIZE)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)






episode_rewards = []

for episode in range(HM_EPISODES):
    start = Blob()
    end = Blob()
    
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (start-end)
        
        if np.random.random() > epsilon:
            
            if(obs>0):
              obs = -1 * obs
            action = np.argmax(q_table[obs])
        else:
            if len(lst[start.x])!=0:
              action = np.random.randint(0, len(lst[start.x]))
            else:
              
              reward = -PENALTY
              break

        curr = start
        start.action(action,lst)
        new = start


        if start.x == end.x:
            reward = REWARD
        else:
            reward = -1*mat[curr.x][new.x]

        new_obs = abs((start-end))
        new_obs = -1 * new_obs
        max_future_q = np.max(q_table[(new_obs)])
        if(obs>0):
            obs = -1 * obs
        current_q = q_table[obs][action]

        if reward == REWARD:
            new_q = REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[end.x][end.x] = d[END_N]  
            env[start.x][start.x] = d[START_N] 
            
            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300))
            cv2.imshow("image" ,np.array(img))  
            if reward == REWARD or reward == -PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        episode_reward += reward
        if reward == REWARD or reward == -PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY