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
        self.target_wz = 0.0
        self.action_scale = 0.25
        self.dt = 0.002

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
        self.scene.build(n_envs = self.batch, env_spacing = (5.0, 5.0))

        # self.robot.set_dofs_kp(torch.tensor([20] * 12),dofs_idx_local = self.joints_local_idx)
        # self.robot.set_dofs_kv(torch.tensor([0.5] * 12),dofs_idx_local = self.joints_local_idx)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
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

        self.action_t = torch.zeros((self.batch, 12))
        self.action_t_1 = torch.zeros((self.batch, 12))
        self.action_t_2 = torch.zeros((self.batch, 12))
        self.previous_joint_velocity = torch.zeros((self.batch, 12))

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
            joints = links.joints
            for joint in joints:
                self.joints_local_idx.append(joint.dof_idx_local)
                low, high = np.rad2deg(joint.dofs_limit)[0]
                self.joints_limit_low.append(low.item())
                self.joints_limit_high.append(high.item())

    def _get_imu_values(self):
        _linear_acc, _angular_vel = self.imu.read()
        """ The arrangement of angular_vel is roll pitch yaw"""
        return torch.tensor(_angular_vel)

    def _get_ypr(self):
        quat_wxyz = self.robot.get_link("base").get_quat()
        quat_xyzw = torch.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], dim = 1)
        rotation = Rotation.from_quat(quat_xyzw.detach().cpu().numpy())
        euler_angles = rotation.as_euler("xyz")
        return torch.tensor(np.array(euler_angles), dtype=torch.float)

    def _calculate_projected_acceleration(self):
        quat = self.robot.get_link("base").get_quat()
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        gx = 2 * (x*z - w*y)
        gy = 2 * (y*z + w*x)
        gz = 1 - 2 * (x*x + y*y)
        return torch.stack([-gx, -gy, -gz], dim=1)
    

    def _calculate_reward(self):
         
        terminated = False
        truncated = False
        sigma = 0.25

        linear_velocity = self.__get_linear_velocity()
        angular_velocity = self._get_imu_values()
        base_height = self.robot.get_link("base").get_pos()[:, 2]
        current_joint_velocity = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)
        torques = self.robot.get_dofs_control_force(dofs_idx_local = self.joints_local_idx)
        projected_acceleration = self._calculate_projected_acceleration()

        vx = linear_velocity[:, 0]
        vy = linear_velocity[:, 1]
        vz = linear_velocity[:, 2]

        reward_alive = 1.0
        linear_velocity_reward = torch.exp(-torch.square(vx - self.target_vx)/sigma)
        reward_angular_velocity_yaw = torch.exp(-torch.square( angular_velocity[:, 2] - self.target_wz)/sigma)
        linear_vel_z_axis_penalty = -torch.square(vz)
        pitch_roll_penalty = -torch.sum(torch.square(angular_velocity[:, :2]), dim=1)
        joint_acceleration_penalty = -torch.sum(torch.square((current_joint_velocity - self.previous_joint_velocity)/self.dt), dim=1)
        action_smoothness_penalty = -torch.sum(torch.square(self.action_t - 2 * self.action_t_1 + self.action_t_2), dim=1)
        height_penalty = -torch.square(self.target_base_height - base_height)
        action_rate_penalty = -torch.sum(torch.square(self.action_t - self.action_t_1), dim=1)
        orientation_penalty = -torch.sum(torch.square(projected_acceleration[:, :2]), dim=1)
        torque_penalty = -torch.sum(torch.square(torques), dim =1)

        # print()
        # print()
        # print(linear_velocity_reward[0], linear_vel_z_axis_penalty[0], pitch_roll_penalty[0], joint_acceleration_penalty[0], action_smoothness_penalty[0],
        #       action_rate_penalty[0], height_penalty[0], orientation_penalty[0], torque_penalty[0])
        total_reward = (
            linear_velocity_reward * 1.0 +
            linear_vel_z_axis_penalty * 2.0 +
            pitch_roll_penalty * 0.05 +
            height_penalty * 2 +
            joint_acceleration_penalty * 2.5e-8 +
            torque_penalty * 0.0001 +
            action_rate_penalty * 0.01 +
            action_smoothness_penalty * 0.01 +
            orientation_penalty * 0.2 +
            reward_alive
        )

        terminated = base_height < 0.15
        total_reward[terminated] -= 2

        return total_reward, terminated
    
    def _get_obs(self):
        """
        https://arxiv.org/pdf/2406.04835v1
        I am using the obs space from here!!!
        """

        linear_velocity = self.__get_linear_velocity()
        angular_velocity = self._get_imu_values()
        joint_positions = self.robot.get_dofs_position(dofs_idx_local = self.joints_local_idx) - self.__initial_positions
        joint_velocities = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)
        previous_action = self.action_t_1
        projected_acceleration = self._calculate_projected_acceleration()

        return torch.cat([
            linear_velocity * 2.0,    #3
            angular_velocity  * 0.25, #3
            projected_acceleration,   #3
            joint_positions ,         #12
            joint_velocities * 0.05,  #12
            previous_action           #12
        ], dim = 1)

    def reset_env_idx(self, envs_ids):

        if len(envs_ids) == 0:
            return
        
        initial_position = torch.tensor(self.__initial_positions).repeat(len(envs_ids), 1)
        self.robot.set_dofs_position(initial_position,dofs_idx_local = self.joints_local_idx,envs_idx = envs_ids)

        zeros = torch.zeros((len(envs_ids), 12), device=self.device)
        self.robot.set_dofs_velocity(zeros, dofs_idx_local=self.joints_local_idx, envs_idx=envs_ids)

        base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(envs_ids), 1)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_ids)

        base_pos = torch.tensor([0, 0, 0.35], device=self.device).repeat(len(envs_ids), 1)
        self.robot.set_pos(base_pos, envs_idx=envs_ids)

        self.action_t_1[envs_ids] = 0
        self.action_t_2[envs_ids] = 0
        self.previous_joint_velocity[envs_ids] =0
        self.episode_len_buffer[envs_ids] = 0
        

    def reset(self, *, seed=None, options=None):
        super().reset()
        # self.reset_env_idx(torchcl.arange(self.batch))
        self.scene.reset()
        observation = self._get_obs()
        info = {}
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
        self.action_t = action.clone()

        target_dof_pos = self.__initial_positions + action * self.action_scale
        # target_dof_pos = self.action_t_1 + action * self.action_scale
        target_dof_pos = torch.clamp(target_dof_pos, self.dof_min, self.dof_max)

        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
        self.scene.step()
       
        reward, terminated = self._calculate_reward()

        self.episode_len_buffer += 1
        truncated = self.episode_len_buffer > 2000

        dones = torch.bitwise_or(terminated, truncated)

        self.action_t_2 = self.action_t_1.clone()
        self.action_t_1 = self.action_t.clone()
        self.previous_joint_velocity = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)

        reset_ids = torch.nonzero(dones).flatten()
        if len(reset_ids) > 0:
            self.reset_env_idx(reset_ids)

        self.timestep += 1
        observation = self._get_obs()
        return observation.detach().cpu().numpy(), reward.detach().cpu().numpy(), dones.detach().cpu().numpy(), {}
    

if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level="warning")
    env = WalkENV(render=True, n_batch=1)
    
    state, info = env.reset()

    while True:
        action = env.action_space.sample()
        state, reward, dones, info = env.step(action)

