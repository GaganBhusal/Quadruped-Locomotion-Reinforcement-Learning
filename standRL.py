import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation

gs.init(backend=gs.cpu, logging_level = "warning")

class StandENV(gym.Env):

    def __init__(self):
        super().__init__()
        
        self.scene = gs.Scene(show_viewer=True)

        plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file='/home/yayy/My/Codeeeeee/Simulators/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf'),
        )

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = self.robot.idx,
                link_idx_local = self.robot.get_link("base").idx_local,
                interpolate = True,
                draw_debug = True
            )
        )

        self._get_internal_info()
        self.scene.build()

        obs_space_low = np.concatenate([
            -np.ones(12),
            -np.ones(12),
            [-1] * 4
        ])

        obs_space_high = np.concatenate([
            np.ones(12),
            np.ones(12),
            [1] * 4
        ])

        self.observation_space = gym.spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(12, ),
            dtype=float
        )

        self.initial_positions = self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).numpy()

    def _get_imu_values(self):
        _linear_acc, _angular_vel = self.imu.read()
        return _linear_acc, _angular_vel

    def _get_ypr(self):
        quat_wxyz = self.robot.get_link("base").get_quat()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        rotation = Rotation.from_quat(quat_xyzw)
        euler_angles = rotation.as_euler("xyz")
        return euler_angles

    def _calculate_reward(self):
         
        terminated = False

        z_coordinate_base = self.robot.get_link("base").get_pos()[2]
        angular_velocity_joints = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx[1:])
        wx, wy, wz = self._get_imu_values()[1]
        roll, yaw, pitch = self._get_ypr()

        reward = (
            z_coordinate_base -
            (wx**2 + wy**2) -
            (roll**2 + pitch*2)
        ) 

        if abs(roll) > 0.4 or abs(pitch) > 0.4:
            reward -= 30
            terminated = True

        return reward.item(), terminated

    
    def _get_obs(self):
        # print(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx[1:]).numpy())

        wx, wy, wz = self._get_imu_values()[1]
        roll, yaw, pitch = self._get_ypr()

        return np.concatenate([
            self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).tolist(),
            self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx).tolist(),
            [wx, wy, roll, pitch],
        ])


    def _get_internal_info(self):
        self.joints_local_idx = []
        self.joints_limit_low = []
        self.joints_limit_high = []
        for links in self.robot.links[1:]:
            print(links)
            joints = links.joints
            for joint in joints:
                self.joints_local_idx.append(joint.dof_idx_local)
                low, high = np.rad2deg(joint.dofs_limit)[0]
                self.joints_limit_low.append(low.item())
                self.joints_limit_high.append(high.item())
                # self.ranges.append(torch.rad2deg(torch.tensor(joint.dofs_limit)))/
                # print(f"Joint Name : {joint.name}, DOF : {joint.n_dofs}, type : {type(joint)}, Axis : {torch.rad2deg(torch.tensor(joint.dofs_limit))}")
        # print(f"Ranges : {self.joints_limit_low}\n{self.joints_limit_high}")

    def reset(self):

        super().reset()
        self.robot.control_dofs_position(
            self.initial_positions, 
            dofs_idx_local = self.joints_local_idx
        )
        observation = self._get_obs()
        info = {}
        self.scene.step()
        return observation, info
        
    def step(self, action):
        self.new_joint_angles = np.add(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx), action)

        self.robot.control_dofs_position(
            self.new_joint_angles, 
            dofs_idx_local = self.joints_local_idx
        )

        reward, terminated = self._calculate_reward()
        observation = self._get_obs()
        truncated = False
        info = {}

        self.scene.step()
        return observation, reward, terminated, truncated, info
    

env = StandENV()
print(env.reset())
print(env.step(action=[0.5] * 12))
