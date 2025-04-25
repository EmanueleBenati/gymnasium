
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

    def __init__(self, render_mode = None, size = 50):
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

        self.window = None
        self.clock = None
        self._current_step = 0
        self.step_size = 1
        
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

        if self.render_mode == "human":
            self._render_frame()

        return observation, info 
    
    def step(self,action):

        # Update agent position
        direction_agent = action * self.step_size
        self._agent_location = np.clip(
            self._agent_location + direction_agent,0,self.size,
        )
        
        # Update target position 
        direction_target = self.np_random.uniform(-1,1,size = 3).astype(np.float32)
        self._target_location = np.clip(
            self._target_location + direction_target, 0, self.size
        )

        #an episode is done if the agent has reached the target or is near a certain threshold (0.5)
        terminated = self._get_info()["distance"] < 1
        
        observation = self._get_obs()
        info = self._get_info()

        if not terminated:

            reward = (- (1 - (1/info["distance"]))) / 10

        else:
            reward = 0
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._render_frame()

        return observation, reward, terminated, False, info 

    def render(self):
        return self._render_frame()

    #Define rendering  via matplotlib 
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((700, 700), DOUBLEBUF | OPENGL)
            pygame.display.set_caption("GridWorld 3D - PyOpenGL")
            glEnable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            gluPerspective(45, 1.0, 0.1,300.0)
            glMatrixMode(GL_MODELVIEW)
            glTranslatef(-self.size / 2, -self.size / 2, -self.size * 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        def draw_cube(pos, color=(1, 1, 1), size=1.5):
            x, y, z = pos
            s = size / 2
            vertices = [
                (x - s, y - s, z - s), (x + s, y - s, z - s),
                (x + s, y + s, z - s), (x - s, y + s, z - s),
                (x - s, y - s, z + s), (x + s, y - s, z + s),
                (x + s, y + s, z + s), (x - s, y + s, z + s),
            ]
            faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]
            glColor3f(*color)
            glBegin(GL_QUADS)
            for face in faces:
                for v in face:
                    glVertex3f(*vertices[v])
            glEnd()

        def draw_axes(length=10):
            glBegin(GL_LINES)
            glColor3f(1, 0, 0)  # X - rosso
            glVertex3f(0, 0, 0)
            glVertex3f(length, 0, 0)
            glColor3f(0, 1, 0)  # Y - verde
            glVertex3f(0, 0, 0)
            glVertex3f(0, length, 0)
            glColor3f(0, 0, 1)  # Z - blu
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, length)
            glEnd()

        def draw_wireframe_cube(size):
            s = size
            glColor3f(0.5, 0.5, 0.5)
            glBegin(GL_LINES)
            for x in [0, s]:
                for y in [0, s]:
                    glVertex3f(x, y, 0)
                    glVertex3f(x, y, s)
            for x in [0, s]:
                for z in [0, s]:
                    glVertex3f(x, 0, z)
                    glVertex3f(x, s, z)
            for y in [0, s]:
                for z in [0, s]:
                    glVertex3f(0, y, z)
                    glVertex3f(s, y, z)
            glEnd()

        draw_axes(length=self.size / 4)
        draw_wireframe_cube(size=self.size)

        # agent = blu, target = rosso
        draw_cube(self._agent_location, color=(0, 0, 1))
        draw_cube(self._target_location, color=(1, 0, 0))

        pygame.display.flip()
        pygame.time.wait(int(1000 / self.metadata["render_fps"]))

        if self.render_mode == "rgb_array":
            return None

    
    def close(self):
        if hasattr(self, "_fig") and self._fig is not None:
            plt.close(self._fig)
            self._fig = None  # Optional: clear the reference

