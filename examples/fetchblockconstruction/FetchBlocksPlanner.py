from rlkit.util.pyhop import pyhop as hop
import numpy as np
from gym import spaces
import copy


extend_goal_threshold = 0.12
height_threshold = 0.57
goal_threshold = 0.05
block_threshold = 0.07
distance_threshold = 0.01
on_table_threshold = 0.45
table_behind = 1.15
table_front = 1.45
table_left = 0.5
table_right = 1.0
on_table_constant = 0.4241327


def achieve_goal(state, goal):
    # Check if goal state reached
    goal_achieved = True
    if len(state.blocks) == 1:
        return [('solve', goal)]
    for block in state.blocks:
        if np.linalg.norm(state.pos[block] - goal.pos[block]) > distance_threshold:
            goal_achieved = False
            break
    if goal_achieved:
        return [('solve', goal)]
    return [('solve', goal)]


def solve(state, goal):
    # Choose next block to place

    # If any on_table_goal is pending, pick that
    on_table_goal_blocks = list(filter(lambda i: goal.on_table[i], goal.blocks))
    for block in on_table_goal_blocks:
        if not state.on_goal[block]:
            return [('move_to_goal', block, goal.pos[block], goal), ('solve', goal)]

    # Pick block on higher level
    not_on_table_goal = list(filter(lambda i: not goal.on_table[i], goal.blocks))
    for block in not_on_table_goal:
        if not state.on_goal[block]:
            # assert blocks below the current block are placed
            goal_pos = goal.pos[block]
            all_good = True
            for g_block in goal.blocks:
                if block == g_block:
                    continue
                if goal.pos[g_block][2] < goal_pos[2] and not state.on_goal[g_block]:
                    all_good = False
                    break
            if all_good:
                return [('move_to_goal', block, goal.pos[block], goal), ('solve', goal)]

    return [('place', block, state.pos[block], goal)]


def move_to_goal(state, block, block_goal, goal):
    return [('place', block, block_goal, goal)]


def clean_up(state, goal):
    # Check any block around the goal location on table
    clean_up_actions = []

    # Ensure blocks are away from all goal locations which are on_table.

    on_table_goal_blocks = list(filter(lambda i: goal.on_table[i], goal.blocks))

    for block in state.blocks:
        maintain_distance = goal_threshold
        if goal.pos[block][2] > height_threshold:
            maintain_distance += extend_goal_threshold
        for g_block in on_table_goal_blocks:
            goal_pos = goal.pos[g_block]
            if np.linalg.norm(state.pos[block][:2] - goal_pos[:2]) < maintain_distance:
                clean_up_actions.append(('move_to_safe', block, goal))
                break

    return clean_up_actions


def go_to(state, block):
    # move arm to the block location
    new_loc = state.pos[block]
    state.arm_position[0] = new_loc[0]
    state.arm_position[1] = new_loc[1]
    return state, [block], []


def pick_up(state, block):
    # pick up the block into the arm
    if state.in_hand is not None:
        return False, None, None
    new_loc = state.pos[block]
    state.arm_position[0] = new_loc[0]
    state.arm_position[1] = new_loc[1]
    state.arm_position[2] = new_loc[2] + 0.05
    state.pos[block] = np.array(new_loc) + [0, 0, 0.5]
    state.on_table[block] = False
    state.in_hand = block
    return state, [block], []


def place(state, move_block, new_loc, goal):
    # place the block at given location
    state.pos[move_block] = np.array(new_loc)
    state.on_goal[move_block] = np.linalg.norm(goal.pos[move_block] - state.pos[move_block]) < distance_threshold
    state.on_table[move_block] = state.pos[move_block][2] < on_table_threshold
    state.in_hand = None
    state.arm_position[0] = new_loc[0]
    state.arm_position[1] = new_loc[1]
    state.arm_position[2] = new_loc[2] + 0.05
    return state


def is_safe_to_move(state, goal, move_block, new_location, lower_threshold=False):
    # Method to verify safety of location x,y for move_block
    on_table_goal_blocks = list(filter(lambda i: goal.on_table[i], goal.blocks))
    maintain_distance = goal_threshold
    if goal.pos[move_block][2] > height_threshold and not lower_threshold:
        maintain_distance += extend_goal_threshold
    for g_block in on_table_goal_blocks:
        goal_pos = goal.pos[g_block]
        if np.linalg.norm(new_location[:2] - goal_pos[:2]) < maintain_distance:
            return False

    # Check 2: Away from existing blocks
    # TODO : Check 3 : No blocks on straight line
    old_location = state.pos[move_block]
    for s_block in state.blocks:
        if s_block == move_block:
            continue
        current_pos = state.pos[s_block]
        if np.linalg.norm(new_location[:2] - current_pos[:2]) < block_threshold:
            return False
    return True


def leave_at_same_location(state, move_block, goal):
    return []


def add_sample_x_location(x):
    def move_to_safe_x_location_dynamic_method(state, move_block, goal):
        new_position = [x, state.pos[move_block][1], on_table_constant]
        if is_safe_to_move(state, goal, move_block, new_position):
            return [('place', move_block, new_position, goal)]
        return False

    move_to_safe_x_location_dynamic_method.__name__ = "move_to_safe_x_location_%s" % (str(x).replace(".", "_"))
    return move_to_safe_x_location_dynamic_method


def add_sample_y_location(y):
    def move_to_safe_y_location_dynamic_method(state, move_block, goal):
        new_position = [state.pos[move_block][0], y, on_table_constant]
        if is_safe_to_move(state, goal, move_block, new_position):
            return [('place', move_block, new_position, goal)]
        return False

    move_to_safe_y_location_dynamic_method.__name__ = "move_to_safe_y_location_%s" % (str(y).replace(".", "_"))
    return move_to_safe_y_location_dynamic_method


def add_sample_location(x, y, lower_threshold=False):
    def move_to_safe_xy_location_dynamic_method(state, move_block, goal):
        new_position = [x, y, on_table_constant]
        if is_safe_to_move(state, goal, move_block, new_position, lower_threshold):
            return [('place', move_block, new_position, goal)]
        return False

    move_to_safe_xy_location_dynamic_method.__name__ = "move_to_safe_location_x_%s_y_%s" % (
        str(x).replace(".", "_"), str(y).replace(".", "_"))
    return move_to_safe_xy_location_dynamic_method


def define_move_to_safe_methods():
    move_x_y_locations = []
    for x in np.arange(table_behind, table_front, 0.05).tolist():
        for y in np.arange(table_left, table_right, 0.05).tolist():
            move_x_y_locations.append(add_sample_location(round(x, 2), round(y, 2)))
    move_x_locations = []
    for x in np.arange(table_behind, table_front, 0.05).tolist():
        move_x_locations.append(add_sample_x_location(round(x, 2)))
    move_y_locations = []
    for y in np.arange(table_left, table_right, 0.05).tolist():
        move_y_locations.append(add_sample_y_location(round(y, 2)))
    method_list_1 = move_x_locations + move_y_locations + move_x_y_locations
    # declare_methods('move', *method_list)
    move_x_y_locations_2 = []
    for x in np.arange(table_behind, table_front, 0.01).tolist():
        for y in np.arange(table_left, table_right, 0.01).tolist():
            move_x_y_locations_2.append(add_sample_location(round(x, 2), round(y, 2), True))
    move_x_locations_2 = []
    for x in np.arange(table_behind, table_front, 0.01).tolist():
        move_x_locations_2.append(add_sample_x_location(round(x, 2)))
    move_y_locations_2 = []
    for y in np.arange(table_left, table_right, 0.01).tolist():
        move_y_locations_2.append(add_sample_y_location(round(y, 2)))
    method_list_2 = move_x_locations_2 + move_y_locations_2 + move_x_y_locations_2
    method_list = method_list_1 + method_list_2 + [leave_at_same_location]
    hop.declare_methods('move_to_safe', *method_list)


def declare_methods_and_operators():
    hop.declare_methods('achieve_goal', achieve_goal)
    hop.declare_methods('clean_up', clean_up)
    hop.declare_methods('solve', solve)
    hop.declare_methods('move_to_goal', move_to_goal)
    define_move_to_safe_methods()
    # print_methods()
    # hop.declare_operators(go_to, pick_up, place)
    hop.declare_operators(place)
    # print_operators()


def get_environment_state(env):
    """
     Converts GYM environment to state.

    :param env: gym env Box World
    :return: State with following attributes

    """
    obs = env.unwrapped._get_obs()
    return get_planner_state(env.unwrapped.num_blocks, obs)


def get_planner_state_from_obs(num_block, obs, dt=distance_threshold):
    state = hop.State('state1')
    state.arm_position = obs[0:3]
    block_features = obs[10:-(3 * (num_block + 1))]
    current_xyz = block_features.reshape((num_block, 15))  # +1 for arm
    state.blocks = [i for i in range(num_block)]
    state.pos = dict((state.blocks[i], current_xyz[i, 0:3]) for i in range(num_block))
    state.on_table = dict((state.blocks[i], current_xyz[i, 2] <= 0.45) for i in range(num_block))

    state.in_hand = None
    goal = hop.Goal('goal1')
    goal_xyz = obs[-(3 * (num_block + 1)):].reshape((num_block + 1, 3))
    goal.blocks = state.blocks.copy()
    goal.pos = dict((goal.blocks[i], goal_xyz[i]) for i in range(num_block))
    goal.on_table = dict((goal.blocks[i], goal_xyz[i, 2] <= on_table_threshold) for i in range(num_block))
    if num_block == 1:
        state.on_goal = dict({0: False})
    else:
        state.on_goal = dict((i, np.linalg.norm(goal.pos[i] - state.pos[i]) < dt) for i in goal.blocks)
    return state, goal


def get_planner_state(num_block, obs):
    state = hop.State('state1')
    current_xyz = obs['achieved_goal'].reshape((num_block + 1, 3))  # +1 for arm
    state.blocks = [i for i in range(num_block)]
    state.pos = dict((state.blocks[i], current_xyz[i].copy()) for i in range(num_block))
    state.on_table = dict((state.blocks[i], current_xyz[i, 2] <= 0.45) for i in range(num_block))
    state.arm_position = obs['achieved_goal'][-3:]
    state.in_hand = None
    goal = hop.Goal('goal1')
    goal_xyz = obs['desired_goal'].reshape((num_block + 1, 3))
    goal.blocks = state.blocks.copy()
    goal.pos = dict((goal.blocks[i], goal_xyz[i]) for i in range(num_block))
    goal.on_table = dict((goal.blocks[i], goal_xyz[i, 2] <= on_table_threshold) for i in range(num_block))
    state.on_goal = dict((i, np.linalg.norm(goal.pos[i] - state.pos[i]) < distance_threshold) for i in goal.blocks)
    return state, goal


class FetchBlocksPlanner:

    def __init__(self, env,
                 observation_key='observation',
                 desired_goal_key='desired_goal',
                 achieved_goal_key='achieved_goal',
                 wait_steps=3,
                 k=3):
        declare_methods_and_operators()
        self.plan = None
        self.goal = None
        self.complete_plan = None
        self.num_blocks = env.unwrapped.num_blocks
        action_dim = 4
        object_dim = 15
        goal_dim = 3
        arm_dim = 10
        self.operator_list = self.get_operators()
        self.k = k
        # This is for buffer
        ob_spaces = None
        ob_spaces = dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(goal_dim * 2,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(goal_dim * 2,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(arm_dim + object_dim,), dtype='float32'),
        )
        self.dims = {'place': (arm_dim, object_dim, goal_dim, action_dim, ob_spaces)}
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.wait = wait_steps
        self._counter = 0
        self.achieved_goal_key = achieved_goal_key

    def set_goal(self, _goal):
        self.goal = _goal

    def get_plan(self, state):
        initial_state, goal_state = get_planner_state_from_obs(self.num_blocks, state)
        return hop.pyhop(initial_state, [('achieve_goal', goal_state)], verbose=0)

    def get_next_operator(self, state_dict):
        state = np.concatenate([state_dict['observation'], state_dict['desired_goal']], axis=0)
        if self.plan is None:
            sub_tasks = self.get_plan(state)
            self._complete_plan = copy.deepcopy(sub_tasks)
            sub_tasks.reverse()
            self.plan = sub_tasks
        if not self.plan:
            self._counter += 1
            if self._counter >= self.wait:
                self._counter = 0
                initial_state, goal_state = get_planner_state_from_obs(self.num_blocks, state, 0.05)
                if not np.all(list(initial_state.on_goal.values())):
                    sub_tasks = hop.pyhop(initial_state, [('solve', goal_state)], verbose=0)
                    sub_tasks.reverse()
                    self.plan = sub_tasks
                    self._complete_plan = self._complete_plan + sub_tasks
                    sub_task = self.plan.pop()
                    return sub_task[0], sub_task[1:]
            return None, None
        sub_task = self.plan.pop()
        return sub_task[0], sub_task[1:]

    def get_operators(self):
        return list(hop.operators.keys())

    def reset(self):
        self.plan = None
        self.goal = None
        self._complete_plan = None
        self._counter = 0

    def get_dim(self, operator):
        return self.dims[operator]

    def get_abstract_state(self, operator, subgoal, obs):
        '''Convert state vector (i.e. obs+goal) to abstract state (i.e. obs+goal)'''
        r = obs.shape[0]
        num_blocks = int((r - 13) / 18)
        assert num_blocks == self.num_blocks, f"Num of blocks in observation ({num_blocks}) " \
                                              f"must match the number of blocks in env ({self.num_blocks})"
        main_block = int(subgoal[0])
        arm_features = obs[0:10]
        block_features = obs[10:-(3 * (self.num_blocks + 1))].reshape((self.num_blocks, 15))
        current_goal = subgoal[1]
        original_goal = obs[-(3 * (num_blocks + 1)):].reshape((num_blocks + 1, 3))
        new_obs = np.append(arm_features, block_features[main_block])
        new_obs = np.append(new_obs, current_goal)
        new_obs = np.append(new_obs, original_goal[-1])  # Appending arm goal position
        return new_obs

    def get_abstract_block_representation(self, obs, subgoal):
        tracked_block = int(subgoal[0])
        arm_features = obs[0:10]
        block_features = obs[10:-(3 * (self.num_blocks + 1))].reshape((self.num_blocks, 15))
        new_obs = np.append(arm_features, block_features[tracked_block])
        return new_obs

    def get_abstract_dict(self, operator, subgoal, obs):
        '''Convert state dictionary to abstract state dictionary'''
        r = obs[self.observation_key].shape[0]
        num_blocks = int((r - 10) / 15)
        assert num_blocks == self.num_blocks, f"Num of blocks in observation ({num_blocks}) " \
                                              f"must match the number of blocks in env ({self.num_blocks})"
        main_block = int(subgoal[0])
        new_obs = {}
        obs_vector = np.hstack((obs[self.observation_key], obs[self.desired_goal_key]))
        arm_features = obs[self.observation_key][0:10]
        original_block_features = obs[self.observation_key][10:].reshape((self.num_blocks, 15))
        original_achieved_goal = obs[self.achieved_goal_key].reshape(
            (self.num_blocks + 1, 3))
        original_desired_goal = obs[self.desired_goal_key].reshape(
            (self.num_blocks + 1, 3))
        current_goal = subgoal[1]
        new_obs[self.observation_key] = np.append(arm_features, original_block_features[main_block])
        new_obs[self.achieved_goal_key] = np.append(original_achieved_goal[main_block], original_achieved_goal[-1])
        new_obs[self.desired_goal_key] = np.append(current_goal, original_desired_goal[-1])
        return new_obs


    def is_terminal(self, operator, subgoal, obs):
        if operator == 'place':
            # obs = np.concatenate([obs_dict['observation'], obs_dict['desired_goal']],axis=0)
            current_goal = subgoal[1]
            tracked_block = int(subgoal[0])
            r = obs.shape[0]
            num_blocks = int((r - 13) / 18)
            assert num_blocks == self.num_blocks, f"Num of blocks in observation ({num_blocks}) " \
                                                  f"must match the number of blocks in env ({self.num_blocks})"
            arm_position = obs[0:3]
            block_features = obs[10:-(3 * (num_blocks + 1))]
            current_xyz = block_features.reshape((num_blocks, 15))
            # goal_xyz = obs[-(3 * (num_blocks + 1)):].reshape((num_blocks + 1, 3))
            if np.linalg.norm(current_goal - current_xyz[tracked_block, 0:3]) < goal_threshold and \
                    np.linalg.norm(arm_position - current_goal) > 0.1:
                return True
        return False

declare_methods_and_operators()

if __name__ == '__main__':
    import time
    import gym
    import fetch_block_construction

    env = gym.make(
        "FetchBlockConstruction_2Blocks_IncrementalReward_DictstateObs_42Rendersize_FalseStackonly_SingletowerCase-v1")
    planner = FetchBlocksPlanner(env)
    for _ in range(100000):
        obs_dict = env.reset()
        planner.reset()
        operator, task = planner.get_next_operator(obs_dict)
        if operator is None:
            planner.reset()
            planner.get_next_operator(obs_dict)
