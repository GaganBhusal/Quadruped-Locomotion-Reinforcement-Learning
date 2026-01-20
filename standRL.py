import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation


class StandENV(gym.Env):

    def __init__(self, render = False, backend = gs.cpu):
        super().__init__()

        self.timestep_low_z_value = 0
        self.timestep = 0

        gs.init(backend=backend, logging_level = "warning")
        self.scene = gs.Scene(show_viewer=render)

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
            low=-np.inf,
            high=np.inf,
            shape=(28, ),
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(12, ),
            dtype=float
        )

        self.__initial_positions = np.deg2rad(
            np.array([
                0, 0, 0, 0, 45, 45, 45, 45, -120, -120, -100, -100
            ])
        )

        self.__joint_ranges = np.deg2rad(
            np.array([
                (-10, 10),
                (-10, 10),
                (-10, 10),
                (-10, 10),

                (25, 70),
                (25, 70),

                (40, 110),
                (40, 110),

                (-130, -70),
                (-130, -70),
                (-130, -70),
                (-130, -70)
            ])
        )

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
        truncated = False

        z_coordinate_base = self.robot.get_link("base").get_pos()[2]
        angular_velocity_joints = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx[1:])
        # print(f"Angular Velocity Joints : {angular_velocity_joints ** 2}")
        wx, wy, wz = self._get_imu_values()[1]
        roll, yaw, pitch = self._get_ypr()

        reward = (
            z_coordinate_base * 2 -
            (wx**2 + wy**2) -
            (roll**2 + pitch*2) - 
            torch.sum(angular_velocity_joints ** 2)
        )

        if z_coordinate_base < 0.16:
            self.timestep_low_z_value += 1
            reward -= 1

        elif z_coordinate_base >= 0.36:
            reward -= 1

        else:
            reward += 2

        if abs(roll) > 1 or abs(pitch) > 1:
            reward -= 30
            terminated = True


        if self.timestep_low_z_value >= 1000:
            reward -= 20
            truncated = True

        if self.timestep > 7000:
            truncated = True

        reward -= self.__out_of_range

        return reward.item(), terminated, truncated

    
    def _get_obs(self):
        # print(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx[1:]).numpy())

        wx, wy, wz = self._get_imu_values()[1]
        roll, yaw, pitch = self._get_ypr()

        return np.concatenate([
            self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).tolist(),
            self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx).tolist(),
            [wx, wy, roll, pitch],
        ])

    def reset(self, *, seed=None, options=None):

        super().reset()
        self.scene.reset()

        self.robot.set_dofs_position(
            self.__initial_positions, 
            dofs_idx_local = self.joints_local_idx
        )

        observation = self._get_obs()
        info = {}
        self.timestep = 0
        self.timestep_low_z_value = 0
        self.scene.step()
        return observation, info
        
    def __clip_angles(self):
        self.__out_of_range = 0
        for i in range(12):
            current_angle = self.new_joint_angles[i]
            current_range = self.__joint_ranges[i]

            if current_angle < current_range[0]:
                self.__out_of_range += abs(current_angle - current_range[0])
            elif current_angle > current_range[1]:
                self.__out_of_range += abs(current_angle - current_range[1])
            
            self.new_joint_angles[i] = np.clip(current_angle, current_range[0], current_range[1])

    def step(self, action):
        self.new_joint_angles = np.add(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx), np.array(action) * 0.5)


        self.__clip_angles()

        self.robot.control_dofs_position(
            self.new_joint_angles, 
            dofs_idx_local = self.joints_local_idx
        )


        reward, terminated, truncated = self._calculate_reward()
        observation = self._get_obs()
        info = {}

        self.timestep += 1
        self.scene.step()
        return observation, reward, terminated, truncated, info
    
if __name__ == "__main__":
    env = StandENV(render=True)
    print(env.reset())
    print(env.step(action=[0.5] * 12))
