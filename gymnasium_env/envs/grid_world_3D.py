
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

    def __init__(self, render_mode = None, size = 100.0):
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

        #initialize env variables
        self._fig = None
        self._ax = None
        self.window = None
        self.clock = None
        self._current_step = 0
        self.step_size = 1
        self._direction_agent = np.zeros(3, dtype=np.float32)
        self._direction_target = np.zeros(3, dtype=np.float32)
        
        # Interactive mode
        plt.ion()
    
    def _is_target_in_cone(self, cone_origin, cone_direction, cone_angle_deg = 20, cone_length=10, cone_min_distance=0):
        """
        Determina se il target si trova all'interno del cono di attacco e anche a una distanza specifica.

        Args:
            cone_origin (np.array): Posizione dell'agente (origine del cono).
            cone_direction (np.array): Direzione del cono (= direzione di movimento dell'agente).
            cone_angle_deg (float): Mezzo angolo di apertura del cono (in gradi).
            cone_length (float): Distanza massima del cono.
            cone_min_distance (float): Distanza minima richiesta (default 0).

        Returns:
            bool: True se il target è dentro al cono **e** nella fascia di distanza, False altrimenti.
        """
        # Vettore dal cono (agente) al target
        vec_to_target = self._target_location - cone_origin
        distance = np.linalg.norm(vec_to_target)

        # Check distanza: il target deve essere nella fascia specificata
        if distance > cone_length or distance < cone_min_distance or distance < 1e-6:
            return False

        # Normalizzazione vettori
        if np.linalg.norm(cone_direction) < 1e-6:
            return False  # direzione nulla = cono non valido

        unit_vec_to_target = vec_to_target / distance
        unit_cone_dir = cone_direction / np.linalg.norm(cone_direction)

        # Confronta angolo tra direzione cono e vettore verso il target
        cos_angle = np.dot(unit_cone_dir, unit_vec_to_target)
        cos_limit = np.cos(np.radians(cone_angle_deg))

        return cos_angle >= cos_limit



    def _plot_attack_cone(self, ax, origin, direction, angle_deg=20, length=10, resolution=20):

        lines = []
        direction = direction / np.linalg.norm(direction)
        angle_rad = np.deg2rad(angle_deg)
        radius = np.tan(angle_rad) * length

        # Crea cerchio base nel sistema z+
        theta = np.linspace(0, 2 * np.pi, resolution)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        circle_z = np.ones_like(theta) * length
        base_circle = np.vstack((circle_x, circle_y, circle_z)).T  # (N,3)

        # Ruota da z+ a direction
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / np.linalg.norm(v)**2)

        rotated_circle = base_circle @ R.T
        cone_base_world = rotated_circle + origin

        # Disegna i lati del cono
        for point in cone_base_world:
            line, = ax.plot(
                [origin[0], point[0]],
                [origin[1], point[1]],
                [origin[2], point[2]],
                color='orange', alpha=0.3
            )
            lines.append(line)

        # Base del cono
        base = np.vstack((cone_base_world, cone_base_world[0]))
        base_line, = ax.plot(base[:, 0], base[:, 1], base[:, 2], color='darkorange', alpha=0.5)
        lines.append(base_line)

        return lines
 


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        distance = np.linalg.norm(self._agent_location - self._target_location, ord=2)
        in_cone = self._is_target_in_cone(
            self._agent_location,
            self._direction_agent,
        )

        return {
            "distance": distance,
            "in_cone": "Target is in the cone" if in_cone else "Target is not in the cone"
        }

    def reset(self, seed=None,options = None):
        

        self.np_random = gym.utils.seeding.np_random(seed)[0]
        # Choose initial agent and target locations 
        self._agent_location = self.np_random.uniform(0,self.size,size = 3).astype(np.float32)
        self._target_location = self.np_random.uniform(0,self.size,size = 3).astype(np.float32)

        self._current_step = 0
        
        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        while np.linalg.norm(self._target_location - self._agent_location) < 50.0:
            self._target_location = self.np_random.uniform(
                0, self.size, size=3).astype(np.float32)
        
        # get the intial obs and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "rgb_array":
            self._render_frame()

        return observation, info 
    
    def step(self,action):

        self._previous_distance = self._get_info()["distance"]

        # Update agent position
        self._direction_agent = action * self.step_size
        self._agent_location = np.clip(
            self._agent_location + self._direction_agent,0,self.size,
        )
        
        # Update target position 
        self._direction_target = self.np_random.uniform(-1,1,size = 3).astype(np.float32) * self.step_size
        self._target_location = np.clip(
            self._target_location + self._direction_target, 0, self.size
        )


        self._current_distance = self._get_info()["distance"]

        # definition of step returns
        observation = self._get_obs()
        info = self._get_info()

        #info['previous_distance'] = self._previous_distance
        #info['current_distance'] = self._current_distance

        # termination condition: if the distance is less than 1.0, the episode is terminated
        terminated = self._current_distance < 1.0
        improvement = self._previous_distance - self._current_distance

        in_cone = self._is_target_in_cone(self._agent_location, self._direction_agent)
        # reward shaping
        if terminated:
            reward = -100.0
        else:
            if in_cone:
                reward = +100 + improvement*10 - self._current_distance/self.size
            else:
                reward = -10 + improvement*10 - self._current_distance/self.size
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._render_frame()

        return observation, reward, terminated, False, info 

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self._fig is None:
            # Prima volta: creo la figura e i 2 punti
            self._fig = plt.figure(figsize = (8,8))
            self._ax = self._fig.add_subplot( projection='3d')
            self._agent_scatter = self._ax.scatter([], [], [], color='blue', s=30, marker= 'o', label = 'Agent')
            self._target_scatter = self._ax.scatter([], [], [], color='red', s=30, marker = '*', label = 'Target')
            self._ax.legend(loc='upper right')
            self._ax.set_xlabel('X')
            self._ax.set_ylabel('Y')
            self._ax.set_zlabel('Z')

            self._distance_line, = self._ax.plot([], [], [], color='gray', linestyle='--')
            self._agent_line, = self._ax.plot([], [], [], color='blue', linestyle='-')
            self._target_line, = self._ax.plot([], [], [], color='red', linestyle='-')


            self._ax.set_xlim(0, self.size)
            self._ax.set_ylim(0, self.size)
            self._ax.set_zlim(0, self.size)

            self._distance_text = self._ax.text2D(0.05, 0.95, "", transform=self._ax.transAxes)
            self._target_in_cone = self._ax.text2D(0.05, 0.90, "", transform=self._ax.transAxes)


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

        # aggiorno linea distanza
        self._distance_line.set_data([self._agent_location[0], self._target_location[0]],
            [self._agent_location[1], self._target_location[1]])
        
        self._distance_line.set_3d_properties(
            [self._agent_location[2], self._target_location[2]])
        
        #print distanza
        distance = self._get_info()["distance"]
        self._distance_text.set_text(f"Distance: {distance:.2f}")

        #print in cone
        target_in_cone = self._get_info()['in_cone']
        self._target_in_cone.set_text(target_in_cone)

        # aggiorno i vettori direzionali
        #agente
        self._agent_line.set_data([self._agent_location[0], self._agent_location[0] + self._direction_agent[0]],
                                  [self._agent_location[1], self._agent_location[1] + self._direction_agent[1]])

        self._agent_line.set_3d_properties(
            [self._agent_location[2], self._agent_location[2] + self._direction_agent[2]])
        
        #target
        self._target_line.set_data([self._target_location[0], self._target_location[0] + self._direction_target[0]],
                                  [self._target_location[1], self._target_location[1] + self._direction_target[1]])
        self._target_line.set_3d_properties(
            [self._target_location[2], self._target_location[2] + self._direction_target[2]])
        
        # aggiorno il cono 
        direction = self._direction_agent
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])  # fallback se fermo

        # Pulisce il cono precedente (se disegnato)
        if hasattr(self, "_cone_lines"):
            for line in self._cone_lines:
                line.remove()
        self._cone_lines = []

        # Chiama la funzione di disegno del cono
        self._cone_lines = self._plot_attack_cone(
            self._ax,
            origin=self._agent_location,
            direction=direction
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

