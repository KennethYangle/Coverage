import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, agent_states, landmark_states):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 4
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.06
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world, agent_states, landmark_states)
        return world

    def reset_world(self, world, agent_states, landmark_states):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        
        for i, agent in enumerate(world.agents):
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_pos = agent_states[i][0]
            agent.state.p_vel = agent_states[i][1]
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_pos = landmark_states[i][0]
            landmark.state.p_vel = landmark_states[i][1]
        
        '''
        for i, agent in enumerate(world.agents):
            if i==0 or i==1 or i==2:
                agent.state.p_pos = np.concatenate((np.random.uniform(-1, 0, 1),np.random.uniform(0,1,1)),axis=0)
            elif i==3 or i==4 or i==5:
                agent.state.p_pos = np.random.uniform(0, +1, world.dim_p)
            elif i==6 or i==7 or i==8:
                agent.state.p_pos = np.random.uniform(-1, 0, world.dim_p)
            else:
                agent.state.p_pos = np.concatenate((np.random.uniform(0, 1, 1),np.random.uniform(-1,0,1)),axis=0)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if i==0 or i==1 or i==2:
                landmark.state.p_pos = np.concatenate((np.random.uniform(-1, 0, 1),np.random.uniform(0,1,1)),axis=0)
            elif i==3 or i==4 or i==5:
                landmark.state.p_pos = np.random.uniform(0, +1, world.dim_p)
            elif i==6 or i==7 or i==8:
                landmark.state.p_pos = np.random.uniform(-1, 0, world.dim_p)
            else:
                landmark.state.p_pos = np.concatenate((np.random.uniform(0, 1, 1),np.random.uniform(-1,0,1)),axis=0)
            landmark.state.p_vel = np.zeros(world.dim_p)
            '''
        '''
        for i, agent in enumerate(world.agents):
            if i==0 or i==1:
                agent.state.p_pos = np.concatenate((np.random.uniform(-1, 0, 1),np.random.uniform(0,1,1)),axis=0)
            elif i==2 or i==3 or i==4:
                agent.state.p_pos = np.random.uniform(0, +1, world.dim_p)
            elif i==5 or i==6 or i==7:
                agent.state.p_pos = np.random.uniform(-1, 0, world.dim_p)
            else:
                agent.state.p_pos = np.concatenate((np.random.uniform(0, 1, 1),np.random.uniform(-1,0,1)),axis=0)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if i==0 or i==1:
                landmark.state.p_pos = np.concatenate((np.random.uniform(-1, 0, 1),np.random.uniform(0,1,1)),axis=0)
            elif i==2 or i==3 or i==4:
                landmark.state.p_pos = np.random.uniform(0, +1, world.dim_p)
            elif i==5 or i==6 or i==7:
                landmark.state.p_pos = np.random.uniform(-1, 0, world.dim_p)
            else:
                landmark.state.p_pos = np.concatenate((np.random.uniform(0, 1, 1),np.random.uniform(-1,0,1)),axis=0)
            landmark.state.p_vel = np.zeros(world.dim_p)
        '''

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        '''
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.05:
                occupied_landmarks += 1
        '''
        '''
        for i, landmark in enumerate(world.landmarks):
            if i==0 or i==1 or i==2:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                min_dists += min(dists)
                rew -= min(dists)
            if min(dists) < 0.05:
                occupied_landmarks += 1
        '''

        '''
        for i, landmark in enumerate(world.landmarks):
            dists = []
            if i==0 or i==1:
                for j, agent in enumerate(world.agents):
                    if j==0 or j==1:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
            elif i==2 or i==3 or i==4:
                for j, agent in enumerate(world.agents):
                    if j==2 or j==3 or j==4:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
            elif i==5 or i==6 or i==7:
                for j, agent in enumerate(world.agents):
                    if j==5 or j==6 or j==7:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
            else:
                for j, agent in enumerate(world.agents):
                    if j==8 or j==9:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
        '''
        for i, landmark in enumerate(world.landmarks):
            dists = []
            if i==0 or i==1 or i==2:
                for j, agent in enumerate(world.agents):
                    if j==0 or j==1 or j==2:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
            elif i==3 or i==4 or i==5:
                for j, agent in enumerate(world.agents):
                    if j==3 or j==4 or j==5:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
            elif i==6 or i==7 or i==8:
                for j, agent in enumerate(world.agents):
                    if j==6 or j==7 or j==8:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
            else:
                for j, agent in enumerate(world.agents):
                    if j==9 or j==10 or j==11:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.05:
                    occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    #rew -= 0.5
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        
        '''
        for i, landmark in enumerate(world.landmarks):
            if i==0 or i==1 or i==2:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                rew -= min(dists)
        '''
        '''
        for i, landmark in enumerate(world.landmarks):
            dists = []
            if i==0 or i==1:
                for j, agent in enumerate(world.agents):
                    if j==0 or j==1:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                rew -= min(dists)
            elif i==2 or i==3 or i==4:
                for j, agent in enumerate(world.agents):
                    if j==2 or j==3 or j==4:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                rew -= min(dists)
            elif i==5 or i==6 or i==7:
                for j, agent in enumerate(world.agents):
                    if j==5 or j==6 or j==7:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                rew -= min(dists)
            else:
                for j, agent in enumerate(world.agents):
                    if j==8 or j==9:
                        dists.append(np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos))))
                rew -= min(dists)
        '''
        '''
        for i, landmark in enumerate(world.landmarks):
            dists = []
            if i==0:
                for j, agent in enumerate(world.agents):
                    if j==0:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==1:
                for j, agent in enumerate(world.agents):
                    if j==1:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==2:
                for j, agent in enumerate(world.agents):
                    if j==2:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==3:
                for j, agent in enumerate(world.agents):
                    if j==3:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))            
            elif i==4:
                for j, agent in enumerate(world.agents):
                    if j==4:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==5:
                for j, agent in enumerate(world.agents):
                    if j==5:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==6:
                for j, agent in enumerate(world.agents):
                    if j==6:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==7:
                for j, agent in enumerate(world.agents):
                    if j==7:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==8:
                for j, agent in enumerate(world.agents):
                    if j==8:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==9:
                for j, agent in enumerate(world.agents):
                    if j==9:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            elif i==10:
                for j, agent in enumerate(world.agents):
                    if j==10:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
            else:
                for j, agent in enumerate(world.agents):
                    if j==11:
                        rew -= np.sqrt(np.sum(np.square(landmark.state.p_pos - agent.state.p_pos)))
        '''
        
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 0.5
                    
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
