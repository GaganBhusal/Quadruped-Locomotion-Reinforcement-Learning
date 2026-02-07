import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation


class WalkENV(gym.Env):

    def __init__(self, render = True, backend = gs.gpu, n_batch = 100, ):
        super().__init__()


        self.device = "cuda"
        self.episode_len_buffer = torch.zeros(n_batch, device=self.device)
        self.timestep = 0
        self.batch = n_batch
        self.reward = 0
        self.target_base_height = 0.3
        self.target_vx = 0.5
        self.action_scale = 0.25

        # gs.init(backend=backend, logging_level = "warning")
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
        self.scene.build(n_envs = self.batch, env_spacing = (10.0, 10.0))

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=-np.inf,
            shape=(45, ),
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(12, ),
            dtype=float
        )

        self.__initial_positions = torch.deg2rad(
            torch.tensor([
                0, 0, 0, 0, 45, 45, 45, 45, -120, -120, -100, -100
            ])
        )

        self.last_action = torch.tensor(self.__initial_positions).clone().repeat(self.batch, 1)

        joint_ranges = torch.deg2rad(
            torch.tensor([
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

        self.dof_min = joint_ranges[:, 0]
        self.dof_max = joint_ranges[:, 1]

        self.__get_linear_velocity()



    def __get_linear_velocity(self):
        vel_body = self.robot.get_link("base").get_vel()
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

    def _get_imu_values(self):
        _linear_acc, _angular_vel = self.imu.read()
        # return _linear_acc, torch.tensor(_angular_vel)
        # print(_angular_vel)
        return torch.tensor(_angular_vel)

    def _get_ypr(self):
        quat_wxyz = self.robot.get_link("base").get_quat()
        # print(quat_wxyz)
        quat_xyzw = torch.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], dim = 1)

        rotation = Rotation.from_quat(quat_xyzw.detach().cpu().numpy())
        # print(rotation)
        euler_angles = rotation.as_euler("xyz")
        # print(euler_/angles)
        return torch.tensor(np.array(euler_angles), dtype=torch.float)

    def _calculate_reward(self):
         
        terminated = False
        truncated = False

        # print(self.robot.get_link("base").get_pos())
        base_height = self.robot.get_link("base").get_pos()[:, 2]
        angular_velocity_joints = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx[1:])
        # print(f"Angular Velocity Joints : {angular_velocity_joints ** 2}")
        # print(self._get_imu_values().shape)
        # wx, wy, wz = self._get_imu_values()[1]
        angular_velocity = self._get_imu_values()
        ypr = self._get_ypr()
        linear_velocity = self.__get_linear_velocity()

        vx = linear_velocity[:, 0]
        vy = linear_velocity[:, 1]
        vz = linear_velocity[:, 2]

        wz = angular_velocity[:, 2]


        reward_lin_vel = torch.exp(
            -torch.square(vx - self.target_vx)/0.25
        )

        reward_ang_vel = torch.exp(
            -torch.square(wz)/0.25
        )

        reward_height = -torch.square(base_height - self.target_base_height)
        reward_action_smooth = -torch.sum(torch.square(self.current_action - self.last_action))
        # print()
        reward_bouncing = -torch.square(vz)

        # reward_z_vel = -torch.square(vz)
        # print(reward_lin_vel, reward_ang_vel, reward_height, reward_bouncing, reward_action_smooth)
        total_reward = (
            4 * reward_lin_vel +
            0.5 * reward_ang_vel +
            0.05* reward_height +
            0.5 * reward_bouncing 
            # 0.005 * reward_action_smooth
        )

        total_reward += 2
        # print(total_reward)
        # total_reward *= 0.01
        # reward = (
        #     self.z_coordinate_base * 2 -
        #     (wx**2 + wy**2) -
        #     (roll**2 + pitch*2) - 
        #     torch.sum(angular_velocity_joints ** 2)
        # )

        # vx = linear_velocity[:, 0]
        # vy = linear_velocity[:, 1]
        # # print(f"\n\n\n{vx}\n\n\n")
        # reward = torch.exp(-2.0 * (vx - 0.6)**2)

        # # Reduce orientation penalty (0.5 instead of 1.0)
        # reward -= 0.5 * torch.sum(torch.square(ypr[:, :2]), dim=1) 

        # # Reduce drift penalty significantly (0.5 instead of 2.0)
        # reward -= 0.5 * torch.abs(vy)



        tilted_mask1 = abs(ypr[:, 0]) > 1
        tilted_mask2 =  abs(ypr[:, 2]) > 1
        # print(tilted_mask2, tilted_mask1)
        tilted_mask = tilted_mask1 | tilted_mask2
        # for i in range(self.batch):
        #     if vx[i] < 0.02:
        #         self.timestep_low_vx_value += 1
        # # if z_coordinate_base < 0.16:
        # #     self.timestep_low_z_value += 1

            # if abs(ypr[:, 0][i]) > 1 or abs(ypr[:, 2][i]) > 1:
            #     reward -= 30
            #     terminated = True

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


        # if self.timestep_low_vx_value >= 1000:
        #     self.reward -= 20
        #     truncated = True
        total_reward[tilted_mask] -= 5

        terminated = tilted_mask
        self.episode_len_buffer += 1
        truncated = self.episode_len_buffer > 2000
        # if self.timestep > 10000:
        #     truncated = True

        # self.reward -= self.__out_of_range
        # print(terminated, truncated)
        return total_reward, terminated, truncated
    
    def _get_obs(self):
        # angular_vel = self._get_imu_values()
        ypr = self._get_ypr()
        # print(
        #     ypr.shape,
        #     self.last_action.shape,
        #     self._get_imu_values().shape,
            
        # )
        return torch.cat([
            self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx) - self.__initial_positions,
            self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx) * 0.05,
            self.__get_linear_velocity() * 2,
            self._get_imu_values() * 0.25,
            ypr,
            self.last_action
        ], dim = 1)

    def reset_env_idx(self, envs_ids):

        if len(envs_ids) == 0:
            return
        
        # print(len(envs_ids))
        initial_position = torch.tensor(self.__initial_positions).repeat(len(envs_ids), 1)
        # initial_position += (torch.rand_like(initial_position) - 0.5) * 0.1



        # print(initial_position.shape)
        # print(len(self.joints_local_idx))
        # print(f"The env idx is {envs_ids}, It must be reset !!!!")
        # print(len(self.joints_local_idx))
        # self.reward[envs_ids] = 0
        self.robot.set_dofs_position(
            initial_position,
            dofs_idx_local = self.joints_local_idx,
            envs_idx = envs_ids
        )
        zeros = torch.zeros((len(envs_ids), 12), device=self.device)
        self.robot.set_dofs_velocity(zeros, dofs_idx_local=self.joints_local_idx, envs_idx=envs_ids)

        base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(envs_ids), 1)
        self.robot.set_quat(
            quat=base_quat, 
            envs_idx=envs_ids
        )

        self.episode_len_buffer[envs_ids] = 0
        

    def reset(self, *, seed=None, options=None):

        super().reset()
        self.scene.reset()

        # self.reset_env_idx(
        #     envs_ids=
        # )

        # self.robot.set_dofs_position(
        #     self.__initial_positions, 
        #     dofs_idx_local = self.joints_local_idx
        # )

        observation = self._get_obs()
        info = {}
        # self.timestep = 0
        # self.episode_len_buffer = 0
        # self.reward = 0
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

        action = torch.tensor(action, dtype=torch.float)
        # print(action.shape)
        target_dof_pos = self.__initial_positions + action * self.action_scale

        # self.new_joint_angles = np.add(
        #     np.array(
        #         self.robot.get_dofs_position(
        #         dofs_idx_local = self.joints_local_idx).detach().cpu().numpy()
        #     ), 
        #     np.array(action) * 0.5
        # )

        target_dof_pos = torch.clamp(
            target_dof_pos, self.dof_min, self.dof_max
        )

        # Clipping the angles if they exceeded the range
        # self.__clip_angles()

        # Set Position 
        self.robot.control_dofs_position(
            target_dof_pos, 
            dofs_idx_local = self.joints_local_idx
        )

        self.scene.step()

        self.current_action = action
        observation = self._get_obs()
        reward, terminated, truncated = self._calculate_reward()
        reward *= 0.01
        self.last_action = action.clone()
        # print(terminated, truncated)
        dones = torch.bitwise_or(terminated, truncated)
        # print(dones)
        reset_ids = torch.nonzero(dones).flatten()
        # print("Error from obs")        
        if len(reset_ids) > 0:
            self.reset_env_idx(reset_ids)

        self.timestep += 1
        # print(observation.shape)

        # print(f"The dones shape is {dones.shape}")
        return observation.detach().cpu().numpy(), reward.detach().cpu().numpy(), dones.detach().cpu().numpy(), {}
    

if __name__ == "__main__":
    env = WalkENV(render=True, n_batch=2)

    state, info = env.reset()

    while True:
        action = env.action_space.sample()
        state, reward, ter, trun, info = env.step(action)

