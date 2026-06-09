from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

from pathlib import Path
import torch


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
import pandas as pd


# In[3]:


class RubiksCube(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],#unused
        "render_fps": 30,#unused
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.n_scramble = 0

        # Example: 4 discrete actions
        self.action_space = spaces.Discrete(12)

        # Example: 1D continuous observation vector of length 5
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(54*6,),#6 3x3 faces in a 1d array. each of which is one of 6 colors represented as an int 0 to 5
            dtype=np.int8, #color
        )

        self.state = []
        self.step_count = 0
        self.faces_completed = 0
        self.is_completed = False

        # Optional render state
        self.window = None
        self.clock = None
        

    def _get_obs(self) -> np.ndarray:
        colors = np.array([int(p[3]) for p in self.state])
        return np.eye(int(colors.max())+1,dtype=np.float64)[colors].flatten()

    def _get_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "faces_completed":self.faces_completed,
            'n_scramble':self.n_scramble,
            #num correct faces
        }

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
            # n_scramble = 26
            ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # solved initial state
        self.state = self.place_points_on_cube_faces()

        #sort points by face then x,y,z
        split_faces = self.split_into_faces(self.state)
        for face in ['top','bottom','left','right','front','back']: 
            if face == 'top':
                state = self.sort_by_first_three(split_faces[face])
            else:
                state = np.concatenate((state,self.sort_by_first_three(split_faces[face])),axis=0)
        self.state = state
        
        # make random moves n times
        for _ in range(self.n_scramble):
            self.move(random.randint(0,11))
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.step_count += 1
        #take action
        self.move(action)
        
        
        reward,terminated = self._get_reward()

        # ---- Termination logic (task-defined) ----
        # terminated,_ = self._check_cube_completion()
        # ------------------------------------------

        # ---- Truncation logic (outside MDP, e.g. time limit) ----
        truncated = self.step_count >= self.max_steps
        # ---------------------------------------------------------

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
        
    def _check_cube_completion(self):
        faces = ['front','back','left','right','top','bottom']
        is_completed = True
        num_faces_completed = 0
        for face in faces:
            face_pts,_ = self.get_face(self.state,face,exclude_sides=True)
            face_colors = [p[3] for p in face_pts]
            if len(set(face_colors)) == 1:
                num_faces_completed += 1
            if len(set(face_colors))> 1:
                is_completed = False
        self.faces_completed = num_faces_completed
        self.is_completed = is_completed
        return is_completed,num_faces_completed
        
    def _get_reward(self,reward_type='absolute'):
        is_completed,faces_completed = self._check_cube_completion()
        if reward_type == 'absolute':
            #only reward for completing cube
            return is_completed * 2 + -0.1, is_completed
        elif reward_type == 'breadcrumb':
            return is_completed * 2 + faces_completed -0.01, is_completed
        # elif too many steps, large negative

    
            
    def render(self,radius=1.0, sphere_alpha=0.2, point_color='red', point_size=50):
        # def plot_points_on_sphere(points, radius=1.0, sphere_alpha=0.2, point_color='red', point_size=50):
        """
        Plot a list of 3D Cartesian coordinates on a sphere.
    
        Parameters
        ----------
        points : list of tuple[float, float, float]
            List of (x, y, z) coordinates.
        radius : float
            Radius of the sphere.
        sphere_alpha : float
            Transparency of the sphere surface.
        point_color : str
            Color of plotted points.
        point_size : int
            Size of plotted points.
        """
        points = self.state
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
        ax.plot_surface(x, y, z, color='lightblue', alpha=sphere_alpha, edgecolor='none')
    
        # Plot input points
        points = np.asarray(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:,3],
                   cmap='viridis',
                   s=point_size)
    
        # Keep aspect ratio equal
        ax.set_box_aspect([1, 1, 1])
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Points on a Sphere")
    
        plt.show()

    def close(self):
        #undo anything opened from self.render
        plt.close()

    def rotate_point_on_sphere(self,point, axis, direction=1):
        """
        Rotate a 3D point on a sphere by +/- 90 degrees around a given axis.
    
        Parameters
        ----------
        point : tuple or array-like
            The (x, y, z) point to rotate.
        axis : tuple or array-like
            The axis of rotation as a 3D vector.
        direction : int
            +1 for +90 degrees, -1 for -90 degrees.
    
        Returns
        -------
        np.ndarray
            Rotated point as a length-3 array.
        """
        color = point[3]
        point = point[:3]
        p = np.asarray(point, dtype=float)
        k = np.asarray(axis, dtype=float)
    
        if np.linalg.norm(k) == 0:
            raise ValueError("axis must be non-zero")
        if direction not in (+1, -1):
            raise ValueError("direction must be +1 or -1")
    
        k = k / np.linalg.norm(k)
        theta = direction * np.pi / 2  # 90 degrees
    
        # Rodrigues' rotation formula
        p_rot = (
            p * np.cos(theta)
            + np.cross(k, p) * np.sin(theta)
            + k * np.dot(k, p) * (1 - np.cos(theta))
        )
    
        return [p_rot[0],p_rot[1],p_rot[2],color]

    def rotate_points_on_sphere(self,points, axis, direction=1):
        return np.array([self.rotate_point_on_sphere(point=p,axis=axis,direction=direction) for p in points])

    def get_face(self,points,face,exclude_sides=False): 
        face = face.lower()
        cut = 0.25
        if exclude_sides:
            cut+=0.25
        if face == 'top':
            face_points = [p for p in points if p[2]>cut]
            remainder = [p for p in points if not p[2]>cut]
        elif face == 'bottom':
            face_points = [p for p in points if p[2]<-cut]
            remainder = [p for p in points if not p[2]<-cut]
        elif face == 'left':
            face_points = [p for p in points if p[0]<-cut]
            remainder = [p for p in points if not p[0]<-cut]
        elif face == 'right':
            face_points = [p for p in points if p[0]>cut]
            remainder = [p for p in points if not p[0]>cut]
        elif face == 'front':
            face_points = [p for p in points if p[1]<-cut]
            remainder = [p for p in points if not p[1]<-cut]
        elif face == 'back':
            face_points = [p for p in points if p[1]>cut]
            remainder = [p for p in points if not p[1]>cut]
        assert len(face_points) == 9 + (12*(exclude_sides==False))
        assert len(remainder) == 45 - (12*(exclude_sides==False))
            
        return np.array(face_points),np.array(remainder)
        
    def move_face(self,points,face,direction):
        '''
        face: one of the following: top, bottom, left, right, front, back
        direction: -1 or 1
        '''
        face = face.lower()
        if face in ['top','bottom']:
            axis = (0,0,1)
        elif face in ['left','right']:
            axis = (1,0,0)
        elif face in ['front','back']:
            axis = (0,1,0)
        points_sub, remainder = self.get_face(points,face)
        assert len(points_sub) == 21
        points_sub = self.rotate_points_on_sphere(points_sub,axis=axis,direction=direction)
        return np.concatenate((points_sub,remainder))

    def sort_by_first_three(self,arr):
        arr = np.asarray(arr)
        
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("arr must have shape (n, 4)")
        
        idx = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
        return arr[idx]
        
    def split_into_faces(self,arr):
        faces = ['top','bottom','left','right','front','back']
        return {face:self.get_face(arr,face=face,exclude_sides=True)[0] for face in faces}

    def move(self,action):
        '''
        Rotate a row or column of the cube
        '''
        
        action_map = {
            0:['top',1],
            1:['top',-1],
            2:['bottom',1],
            3:['bottom',-1],
            4:['left',1],
            5:['left',-1],
            6:['right',1],
            7:['right',-1],
            8:['front',1],
            9:['front',-1],
            10:['back',1],
            11:['back',-1],
        }
        points=self.state
        
        choice = action_map[action]
        
        #sort by face, then by x,y,z, then concat 
        moved_faces = self.move_face(points=points,face=choice[0],direction=choice[1])
        split_faces = self.split_into_faces(moved_faces)
        for face in ['top','bottom','left','right','front','back']: 
            if face == 'top':
                state = self.sort_by_first_three(split_faces[face])
            else:
                state = np.concatenate((state,self.sort_by_first_three(split_faces[face])),axis=0)
        self.state = state
        
    def place_points_on_cube_faces(self,grid_range=0.5):
        """Place 9 points on each of the 6 cube-like faces of the sphere.
           Use `grid_range` to control how tightly the points are clustered (0.1 to 1.0)."""
    
        points = []
        radius = 1.0
        center = (0,0,0)
        # Grid from -grid_range to grid_range in 3 steps (local coordinates)
        steps = np.linspace(-grid_range, grid_range, 3)
        
        # For each face (x+, x-, y+, y-, z+, z-)
        for face in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
            for a in steps:
                for b in steps:
                    # Get local coordinates on the cube face
                    if face == 'x+':
                        color = 0
                        x, y, z = 1, a, b
                    elif face == 'x-':
                        # color = 7 #i want to do 7 so colors are different enough, but for now i am lazily using these values for onehot as well
                        color = 1
                        x, y, z = -1, a, b
                    elif face == 'y+':
                        color = 2
                        x, y, z = a, 1, b
                    elif face == 'y-':
                        color = 3
                        x, y, z = a, -1, b
                    elif face == 'z+':
                        color = 4
                        x, y, z = a, b, 1
                    elif face == 'z-':
                        color = 5
                        x, y, z = a, b, -1
                    
                    # Normalize to project onto the sphere
                    norm = np.linalg.norm([x, y, z])
                    x_norm = x / norm
                    y_norm = y / norm
                    z_norm = z / norm
                    
                    # Scale by radius and shift to center
                    x_global = radius * x_norm + center[0]
                    y_global = radius * y_norm + center[1]
                    z_global = radius * z_norm + center[2]
                    
                    points.append((x_global, y_global, z_global,color))
        return points