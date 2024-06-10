from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class MG_5(RoomGrid):
    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.obj_type = obj_type
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, [obj_type]],
        )

        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            mission_space=mission_space,
            room_size=room_size,
            num_rows=num_rows,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

        self.is_goal = True
    
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