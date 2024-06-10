from __future__ import annotations
import cv2
import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball

class MG_2(RoomGrid):
    def __init__(self, max_steps = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 5
        if max_steps is None:
            max_steps = 16 * room_size**2

        
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

        self.is_goal = False

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box", color="blue")
        
        # Make sure the two rooms are directly connected by a locked door
        door_1, pos_1 = self.add_door(0, 0, 0, locked=True, color="red")

        # Block the door with a ball
        color = "blue"
        self.grid.set(pos_1[0] - 1, pos_1[1], Ball(color))

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door_1.color)

        self.add_distractors(0, 0, 1)

        self.add_distractors(1, 0, 1)
        
        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"
    
    def reset(self, seed, options = None):
        self.is_goal = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = 1
                terminated = True
                self.is_goal = True

        return obs, reward, terminated, truncated, info
    
    def get_agent_position(self):
        return self.agent_pos


class MG_2_noisy_TV(RoomGrid):
    def __init__(self, max_steps = None,  **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 5
        if max_steps is None:
            max_steps = 16 * room_size**2
        
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box", color="blue")
        
        # Make sure the two rooms are directly connected by a locked door
        door_1, pos_1 = self.add_door(0, 0, 0, locked=True, color="red")

        # Block the door with a ball
        color = "blue"
        self.grid.set(pos_1[0] - 1, pos_1[1], Ball(color))

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door_1.color)

        self.add_distractors(0, 0, 1)

        self.add_distractors(1, 0, 1)
        
        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"
    
    def reset(self, seed, options = None):
        self.is_goal = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.done:
            info = {}
            obs = {"image":np.random.randint(0, 256, size=(7, 7, 3), dtype=np.uint8), "direction": self.agent_dir, "mission": self.mission}


        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = 1
                terminated = True
                self.is_goal = True

        return obs, reward, terminated, truncated, info
    
    def get_agent_position(self):
        return self.agent_pos
    


class MG_2_noisy_TV_v2(RoomGrid):
    def __init__(self, max_steps = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 5
        if max_steps is None:
            max_steps = 16 * room_size**2
        
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

        self.noisy_list = [False, True, True, True, True]

        self.is_goal = False

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box", color="blue")
        
        # Make sure the two rooms are directly connected by a locked door
        door_1, pos_1 = self.add_door(0, 0, 0, locked=True, color="red")

        # Block the door with a ball
        color = "blue"
        self.grid.set(pos_1[0] - 1, pos_1[1], Ball(color))

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door_1.color)

        self.add_distractors(0, 0, 1)

        self.add_distractors(1, 0, 1)
        
        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"
    
    def reset(self, seed, options = None):
        self.is_goal = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        random_number =  np.random.randint(0, 5)

        if action == self.actions.done and self.noisy_list[random_number]:
            info = {}
            obs = {"image":np.random.randint(0, 256, size=(7, 7, 3), dtype=np.uint8), "direction": self.agent_dir, "mission": self.mission}


        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = 1
                terminated = True
                self.is_goal = True

        return obs, reward, terminated, truncated, info
    
    def get_agent_position(self):
        return self.agent_pos