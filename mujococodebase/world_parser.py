import logging
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.play_mode import PlayModeEnum

logger = logging.getLogger()

class WorldParser:
    def __init__(self, agent):
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent

    def parse(self, message: str) -> None:
        perception_dict: dict = self.__sexpression_to_dict(message)

        world = self.agent.world

        # Game parse

        if world.is_left_team is None:
            world.is_left_team = (
                True
                if perception_dict["GS"]["tl"] == world.team_name
                else False if perception_dict["GS"]["tr"] == world.team_name else None
            )

        world.playmode = PlayModeEnum.get_playmode_from_string(
            playmode=perception_dict["GS"]["pm"], is_left_team=world.is_left_team
        )

        world.game_time = perception_dict["GS"]["t"]
        world.score_left = perception_dict["GS"]["sl"]
        world.score_right = perception_dict["GS"]["sr"]

        left_team_name: str = perception_dict["GS"].get("tl", None)
        right_team_name: str = perception_dict["GS"].get("tr", None)
        if left_team_name and right_team_name:
            world.their_team_name = (
                right_team_name if world.is_left_team else left_team_name
            )

        world.last_server_time = world.server_time
        world.server_time = perception_dict["time"]["now"]

        # Robot parse

        robot = self.agent.robot

        robot.motor_positions = {h["n"]: h["ax"] for h in perception_dict["HJ"]}

        robot.motor_speeds = {h["n"]: h["vx"] for h in perception_dict["HJ"]}

        world._global_cheat_position = np.array(perception_dict["pos"]["p"])

        # changes quaternion from (w, x, y, z) to (x, y, z, w) 
        robot._global_cheat_orientation = np.array(perception_dict["quat"]["q"])
        robot._global_cheat_orientation = robot._global_cheat_orientation[[1, 2, 3, 0]]

        # flips 180 deg considering team side
        try:
            if not world.is_left_team:
                world._global_cheat_position[:2] = -world._global_cheat_position[:2]

                global_rotation = R.from_quat(robot._global_cheat_orientation)
                yaw180 = R.from_euler('z', 180, degrees=True)
                fixed_rotation = yaw180 * global_rotation
                robot._global_cheat_orientation = fixed_rotation.as_quat()

            # updates global orientation
            euler_angles_deg = R.from_quat(robot._global_cheat_orientation).as_euler('xyz', degrees=True)
            robot.global_orientation_euler = np.array([MathOps.normalize_deg(axis_angle) for axis_angle in euler_angles_deg])
            robot.global_orientation_quat = robot._global_cheat_orientation
            world.global_position = world._global_cheat_position
        except:
            logger.exception(f'Failed to rotate orientation and position considering team side')
            
        robot.gyroscope = np.array(perception_dict["GYR"]["rt"])

        robot.accelerometer = np.array(perception_dict["ACC"]["a"])

        world.is_ball_pos_updated = False

        # Vision parse
        if 'See' in perception_dict:
            
            for seen_object in perception_dict['See']:
                obj_type = seen_object['type']
                
                if obj_type == 'B': # Ball
                    
                    polar_coords = np.array(seen_object['pol'])
                    local_cartesian_3d = MathOps.deg_sph2cart(polar_coords)
                    
                    world.ball_pos = MathOps.rel_to_global_3d(
                        local_pos_3d=local_cartesian_3d,
                        global_pos_3d=world.global_position,
                        global_orientation_quat=robot.global_orientation_quat
                    )
                    world.is_ball_pos_updated = True
                
                elif obj_type == "P":
                    
                    team = seen_object.get('team')
                    player_id = seen_object.get('id')
                    
                    if team and player_id is not None:
                        if (team == world.team_name):
                            player = world.our_team_players[player_id-1]
                        else:
                            player = world.their_team_players[player_id-1]
                        
                        objects = [seen_object.get('head'), seen_object.get('l_foot'), seen_object.get('r_foot')]

                        seen_objects = [object for object in objects if object]

                        if seen_objects:

                            local_cartesian_seen_objects = [MathOps.deg_sph2cart(object) for object in seen_objects]

                            approximated_centroid = np.mean(local_cartesian_seen_objects, axis=0)
 
                            player.position = MathOps.rel_to_global_3d(
                                local_pos_3d=approximated_centroid,
                                global_pos_3d=world.global_position,
                                global_orientation_quat=robot._global_cheat_orientation
                            )
                            player.last_seen_time = world.server_time
                        
                elif obj_type:
                    
                    polar_coords = np.array(seen_object['pol'])
                    world.field.field_landmarks.update_from_perception(
                        landmark_id=obj_type,
                        landmark_pos=polar_coords
                    )

    def __sexpression_to_dict(self, sexpression: str) -> dict:
        """
        Parses a sensor data string of nested parenthesis groups into a structured dictionary.
        Repeated top-level tags are aggregated into lists.
        """

        def split_top_level(s: str):
            """Return a list of substrings that are top-level parenthesized groups."""
            groups = []
            depth = 0
            start = None
            for i, ch in enumerate(s):
                if ch == '(':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and start is not None:
                        groups.append(s[start:i+1])
                        start = None
            return groups

        result = {}

        top_groups = split_top_level(sexpression)

        for grp in top_groups:
            m = re.match(r'^\((\w+)\s*(.*)\)$', grp, re.DOTALL)
            if not m:
                continue
            tag = m.group(1)
            inner = m.group(2).strip()

            if tag == "See":
                see_items = []
                subs = split_top_level(inner)
                
                for sub in subs:
                    sm = re.match(r'^\((\w+)\s*(.*)\)$', sub, re.DOTALL)
                    if not sm:
                        continue
                    obj_type = sm.group(1)
                    inner2 = sm.group(2)

                    if obj_type == "P":  # Player
                        player_data = {"type": "P"}
                        team_m = re.search(r'\(team\s+([^)]+)\)', inner2)
                        if team_m:
                            player_data["team"] = team_m.group(1)
                        id_m = re.search(r'\(id\s+([^)]+)\)', inner2)
                        if id_m:
                            try:
                                player_data["id"] = int(id_m.group(1))
                            except ValueError:
                                player_data["id"] = id_m.group(1)

                        parts = re.findall(r'\((\w+)\s*\(pol\s+([-0-9.\s]+)\)\)', inner2)
                        for part_name, pol_str in parts:
                            pol_vals = [float(x) for x in pol_str.strip().split()]
                            player_data[part_name] = pol_vals

                        see_items.append(player_data)
                        continue

                    # Generic
                    pol_m = re.search(r'\(pol\s+([-0-9.\s]+)\)', inner2)
                    vals = [float(x) for x in pol_m.group(1).strip().split()] if pol_m else []
                    see_items.append({"type": obj_type, "pol": vals})

                result.setdefault("See", []).extend(see_items)
                continue

            # Generic parse for other tags (time, GS, quat, pos, HJ, ...)
            group = {}
            children = split_top_level(inner)
            if children: # (key val1 val2)
                for child in children:
                    im = re.match(r'^\(\s*(\w+)\s+([^)]+)\)$', child.strip(), re.DOTALL)
                    if not im:
                        continue
                    key = im.group(1)
                    vals = im.group(2).strip().split()
                    parsed = []
                    for t in vals:
                        try:
                            parsed.append(float(t))
                        except ValueError:
                            parsed.append(t)
                    group[key] = parsed[0] if len(parsed) == 1 else parsed
            else:
                # search pairs (key vals...)
                items = re.findall(r"\(\s*(\w+)((?:\s+[^()]+)+)\)", inner)
                for key, vals in items:
                    tokens = vals.strip().split()
                    parsed_vals = []
                    for t in tokens:
                        try:
                            parsed_vals.append(float(t))
                        except ValueError:
                            parsed_vals.append(t)
                    # Single value vs. list
                    group[key] = parsed_vals[0] if len(parsed_vals) == 1 else parsed_vals

            # Merge into result, handling repeated tags as lists
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(group)
                else:
                    result[tag] = [result[tag], group]
            else:
                result[tag] = group

        return result
