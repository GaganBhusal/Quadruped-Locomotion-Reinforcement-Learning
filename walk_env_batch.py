import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation


class WalkENV(gym.Env):

    def __init__(self, render = True, backend = gs.gpu, n_batch = 100):
        super().__init__()

        self.timestep_low_vx_value = 0
        self.timestep = 0
        self.batch = n_batch
        self.reward = 0

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
        self.scene.build(n_envs = self.batch, env_spacing = (3.0, 3.0))

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
            high=-np.inf,
            shape=(29, ),
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
        self.__get_linear_velocity()



    def __get_linear_velocity(self):
        vel_body = self.robot.get_link("base").get_vel()
        # print(len(vel_body))
        return torch.tensor(vel_body)

    def _get_internal_info(self):
        self.joints_local_idx = []
        self.joints_limit_low = []
        self.joints_limit_high = []
        for links in self.robot.links[1:]:
            # print(links)
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
        return _linear_acc, torch.tensor(_angular_vel)

    def _get_ypr(self):
        quat_wxyz = self.robot.get_link("base").get_quat()
        # print(quat_wxyz)
        quat_xyzw = torch.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], dim = 1)
        rotation = Rotation.from_quat(quat_xyzw.detach().cpu().numpy())
        euler_angles = rotation.as_euler("xyz")
        return torch.tensor(euler_angles)

    def _calculate_reward(self):
         
        terminated = False
        truncated = False

        self.z_coordinate_base = self.robot.get_link("base").get_pos()[2]
        angular_velocity_joints = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx[1:])
        # print(f"Angular Velocity Joints : {angular_velocity_joints ** 2}")
        # print(self._get_imu_values().shape)
        # wx, wy, wz = self._get_imu_values()[1]
        
        ypr = self._get_ypr()
        linear_velocity = self.__get_linear_velocity()

        # reward = (
        #     self.z_coordinate_base * 2 -
        #     (wx**2 + wy**2) -
        #     (roll**2 + pitch*2) - 
        #     torch.sum(angular_velocity_joints ** 2)
        # )
        vx = linear_velocity[:, 0]
        self.reward += torch.exp(-1.5 * (vx - 1)**2)

        for i in range(self.batch):
            if vx[i] < 0.02:
                self.timestep_low_vx_value += 1
        # if z_coordinate_base < 0.16:
        #     self.timestep_low_z_value += 1

            if abs(ypr[:, 0][i]) > 1 or abs(ypr[:, 2][i]) > 1:
                reward -= 30
                terminated = True

        # if vx<0.02:
        #     self.timestep_low_vx_value += 1
        #     self.reward -= 5
        # if self.z_coordinate_base < 0.23:
        #     self.timestep_low_z_value += 1
        #     reward -= 10
            # print("here")

        # elif self.z_coordinate_base >= 0.36:
        #     reward -= 1

        # else:
        #     reward += 2

        # if abs(ypr[:, 2]) > 0.5 or abs(ypr[:, 1]) > 0.5:
        #     self.reward -= 30
        #     terminated = True


        if self.timestep_low_vx_value >= 1000:
            self.reward -= 20
            truncated = True

        if self.timestep > 10000:
            truncated = True

        self.reward -= self.__out_of_range

        return self.reward, terminated, truncated

    
    def _get_obs(self):
        # print(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx[1:]).numpy())
        # print(self._get_imu_values())
        angular_vel = self._get_imu_values()[1]
        # print(angular_vel[:, 0])
        ypr = self._get_ypr()
        # print(angular_vel.shape, ypr.shape)


        # print(f"Printing shapes \n ------------------------\n\n{self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).shape}\n{self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx).shape}\n{torch.stack([angular_vel[:, [0, 1]]]).shape}\n{torch.stack([ypr[:, [0, 2]]]).shape}")
        
        #         return np.concatenate([
        #     dofs_position,     #12
        #     dofs_velocity,     #12
        #     self._get_imu_values()[1],      #3
        #     self.__get_linear_velocity(),        #3
        #     [roll, pitch],      #2
        # ])
        
        return torch.cat([
            self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx),
            self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx),
            self.__get_linear_velocity(),
            # torch.stack([angular_vel[:, [0, 1]]]).squeeze(0), 
            torch.stack([ypr[:, [0, 2]]]).squeeze(0),
        ], dim = 1)

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
        self.timestep_low_vx_value = 0
        self.reward = 0
        self.scene.step()
        return observation.detach().cpu().numpy(), info
        
    def __clip_angles(self):
        self.__out_of_range = 0
        for i in range(self.batch):
            for j in range(12):
                current_angle = self.new_joint_angles[i][j]
                # print(current_angle.shape)
                current_range = self.__joint_ranges[j]
                # print(current_range.shape)

                if current_angle < current_range[0]:
                    self.__out_of_range += abs(current_angle - current_range[0])
                elif current_angle > current_range[1]:
                    self.__out_of_range += abs(current_angle - current_range[1])
                
                self.new_joint_angles[i][j] = np.clip(current_angle, current_range[0], current_range[1])

    def step(self, action):
        # print(action)
        # print(action.shape)
        # print(type(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).detach().cpu().numpy()))
        self.new_joint_angles = np.add(np.array(self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx).detach().cpu().numpy()), np.array(action) * 0.5)


        self.__clip_angles()

        self.robot.control_dofs_position(
            self.new_joint_angles, 
            dofs_idx_local = self.joints_local_idx
        )


        reward, terminated, truncated = self._calculate_reward()
        observation = self._get_obs()
        # print("Error from obs")
        info = {}

        self.timestep += 1
        self.scene.step()
        # print(observation.shape)
        return observation.detach().cpu().numpy(), reward.detach().cpu().numpy(), terminated, truncated, info
    

if __name__ == "__main__":
    env = StandENV(render=True)
    print(env.reset())
    print(env.step(action=[0.5] * 12))
