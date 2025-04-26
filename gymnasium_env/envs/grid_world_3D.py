
# Building a 3D environment with conitnuous action and obs spaces

import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import pygame 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import glutInit, glutSolidCube, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12

class GridWorld3DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, size = 100):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent":spaces.Box(0, size , shape=(3,), dtype=np.float32),
                "target":spaces.Box(0, size , shape=(3,), dtype=np.float32),
            }

        )
        self.action_space = spaces.Box(
            low = -1.0, high = 1.0,shape = (3,), dtype = np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self._fig = None
        self._ax = None
        self.window = None
        self.clock = None
        self._current_step = 0
        self.step_size = 2
        
        self._fig = None
        # Interactive mode
        plt.ion()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=2
            )
        }

    def reset(self, seed=None,options = None):
        
        self.np_random = gym.utils.seeding.np_random(seed)[0]
        # Choose initial agent and target locations 
        self._agent_location = self.np_random.uniform(0,self.size,size = 3).astype(np.float32)
        self._target_location = self.np_random.uniform(0,self.size,size = 3).astype(np.float32)

        self._current_step = 0
        
        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        while np.linalg.norm(self._target_location - self._agent_location) < 20.0:
            self._target_location = self.np_random.uniform(
                0, self.size, size=3).astype(np.float32)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "rgb_array":
            self._render_frame()

        return observation, info 
    
    def step(self,action):

        self._previous_distance = self._get_info()["distance"]
        # Update agent position
        direction_agent = action * self.step_size
        self._agent_location = np.clip(
            self._agent_location + direction_agent,0,self.size,
        )
        self._current_distance = self._get_info()["distance"]
        
        # Update target position 
        
        direction_target = self.np_random.uniform(-1,1,size = 3).astype(np.float32)
        
        self._target_location = np.clip(
            self._target_location + direction_target, 0, self.size
        )
        
        #an episode is done if the agent has reached the target or is near a certain threshold (1)

        terminated = self._current_distance < 1.0

        observation = self._get_obs()
        info = self._get_info()
        info['previous_distance'] = self._previous_distance
        info['current_distance'] = self._current_distance

        if not terminated:
            if self._previous_distance > self._current_distance:
                reward = -0.1
            else:
                reward = -0.9
        else:
            reward = 1.0
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._render_frame()

        return observation, reward, terminated, False, info 

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self._fig is None:
            # Prima volta: creo la figura e i 2 punti
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot( projection='3d')
            self._agent_scatter = self._ax.scatter([], [], [], color='blue', s=100)
            self._target_scatter = self._ax.scatter([], [], [], color='red', s=100)

            self._ax.set_xlim(0, self.size)
            self._ax.set_ylim(0, self.size)
            self._ax.set_zlim(0, self.size)

         # Aggiorno SOLO la posizione dei punti
        self._agent_scatter._offsets3d = (
            [self._agent_location[0]],
            [self._agent_location[1]],
            [self._agent_location[2]],
        )
        self._target_scatter._offsets3d = (
            [self._target_location[0]],
            [self._target_location[1]],
            [self._target_location[2]],
        )

        self._fig.canvas.draw()


        # converte il canvas matplotlib in array RGB
        frame = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))

        if self.render_mode == "human":
            plt.pause(0.25)
            return None
        elif self.render_mode == "rgb_array":
            return frame


    def close(self):
        if hasattr(self, "_fig") and self._fig is not None:
            plt.close(self._fig)
            self._fig = None  # Optional: clear the reference

