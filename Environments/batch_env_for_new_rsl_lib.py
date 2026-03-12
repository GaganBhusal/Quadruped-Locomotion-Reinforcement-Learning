import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
import tensordict

class WalkENV(gym.Env):

    def __init__(self, render = True, backend = gs.gpu, num_envs = 100, device="cuda"):
        super().__init__()


        self.device = device
        self.num_envs = num_envs
        self.max_episode_length = 1000

        self.target_base_height = 0.3
        self.target_vx = 1.0
        self.action_scale = 0.25

        self.num_obs = 45
        self.num_actions = 12
        self.cfg = {}
        # self.max_episodes_length = 

        # self.episode_len_buffer = torch.zeros(n_batch, device=self.device)
        # self.timestep = 0
        # self.batch = n_batch
        # self.reward = 0
        # self.target_base_height = 0.3
        # self.target_vx = 1.0
        # self.target_wz = 0.0

        self.dt = 0.02

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
        self.scene.build(n_envs = self.num_envs, env_spacing = (5.0, 5.0))

        self.robot.set_dofs_kp(torch.tensor([20] * 12),dofs_idx_local = self.joints_local_idx)
        self.robot.set_dofs_kv(torch.tensor([0.5] * 12),dofs_idx_local = self.joints_local_idx)
        
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
                0, 0, 0, 0, 45, 45, 57, 57, -85, -85, -85, -85
            ])
        )

        self.actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros((self.num_envs, 12))

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


        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)


        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)


        self.__get_linear_velocity()



    def __get_linear_velocity(self):
        vel_body = self.robot.get_link("base").get_vel()
        return torch.tensor(vel_body)

    def _get_internal_info(self):
        self.joints_local_idx = []
        for links in self.robot.links[1:]:
            joints = links.joints
            for joint in joints:
                self.joints_local_idx.append(joint.dof_idx_local)

    def _get_imu_values(self):
        _linear_acc, _angular_vel = self.imu.read()
        """ The arrangement of angular_vel is roll pitch yaw"""
        return _angular_vel

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
    

    def _compute_reward(self):
        # Get state info
        base_pos = self.robot.get_link("base").get_pos()
        base_vel = self.robot.get_link("base").get_vel()
        base_ang_vel = self._get_imu_angular_vel()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx)
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)

        # Linear velocity in body frame
        base_quat = self.robot.get_link("base").get_quat()
        inv_base_quat = self._inv_quat(base_quat)
        base_lin_vel_body = self._transform_by_quat(base_vel, inv_base_quat)

        vx = base_lin_vel_body[:, 0]
        vz = base_lin_vel_body[:, 2]
        base_height = base_pos[:, 2]

        # Tracking error
        lin_vel_error = torch.square(vx - self.target_vx)
        # Penalize vertical velocity
        lin_vel_z_penalty = torch.square(vz)
        # Penalize base height error
        height_error = torch.square(base_height - self.target_base_height)
        # Action rate penalty
        action_rate_penalty = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        # Joint velocity penalty (similar to base example's "similar_to_default"?)
        # Actually base example uses joint position difference, but we'll keep joint velocity smoothness
        # This is optional – we can keep it small
        joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=1)
        # Torque penalty (optional)
        torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx)
        torque_penalty = torch.sum(torch.square(torques), dim=1)

        # Rewards (scaled appropriately – we'll follow base's style)
        reward_tracking = torch.exp(-lin_vel_error / 0.25)          # sigma = 0.25
        reward = (
            + 1.0 * reward_tracking
            - 1.0 * lin_vel_z_penalty
            - 50.0 * height_error
            - 0.005 * action_rate_penalty
            - 0.00001 * joint_vel_penalty
            - 0.0001 * torque_penalty
        )

        # Termination conditions
        # Convert quaternion to euler angles for termination check
        euler = self._quat_to_euler(base_quat)   # (roll, pitch, yaw)
        terminated = (
            (torch.abs(euler[:, 0]) > 0.35) |    # roll > 20 deg
            (torch.abs(euler[:, 1]) > 0.35) |    # pitch > 20 deg
            (base_height < 0.13)
        )

        return reward, terminated

    

    def get_observations(self):
        """Return current observations and extras."""
        return tensordict.TensorDict({"obs": self.obs_buf})

    def get_privileged_observations(self):
        """Return privileged observations (None if not used)."""
        return None

    def _get_obs(self):
        """
        https://arxiv.org/pdf/2406.04835v1
        I am using the obs space from here!!!
        """
                
        base_vel = self.robot.get_link("base").get_vel()
        base_ang_vel = self._get_imu_angular_vel()
        base_quat = self.robot.get_link("base").get_quat()
        inv_base_quat = self._inv_quat(base_quat)
        base_lin_vel_body = self._transform_by_quat(base_vel, inv_base_quat)
        projected_gravity = self._get_projected_gravity()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx) - self.default_dof_pos
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)

        # Scale observations as in base example
        obs = torch.cat([
            base_lin_vel_body * 2.0,                # 3
            base_ang_vel * 0.25,                     # 3
            projected_gravity,                       # 3
            dof_pos,                                  # 12
            dof_vel * 0.05,                           # 12
            self.actions,                             # 12
        ], dim=1)
        return obs

    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset to default joint positions
        self.robot.set_dofs_position(
            self.default_dof_pos.repeat(len(envs_idx), 1),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )
        # Zero velocity
        self.robot.set_dofs_velocity(
            torch.zeros((len(envs_idx), 12), device=self.device),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )
        # Base position and orientation
        base_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(envs_idx), 1)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_idx)
        base_pos = torch.tensor([0, 0, 0.42], device=self.device).repeat(len(envs_idx), 1)
        self.robot.set_pos(base_pos, envs_idx=envs_idx)

        # Reset episode counters
        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0
        

    def reset(self):
        # Reset all envs
        all_idx = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_idx)
        self.scene.reset()
        self.obs_buf = self._get_obs()
        return {"obs": self.obs_buf}, {}
        
    def step(self, action):
        action = torch.tensor(action, dtype=torch.float)
        self.action_t = action.clone()

        target_dof_pos = self.__initial_positions + action * self.action_scale
        # target_dof_pos = self.action_t_1 + action * self.action_scale
        target_dof_pos = torch.clamp(target_dof_pos, self.dof_min, self.dof_max)

        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
        for _ in range(10):
            self.scene.step()
       
        reward, terminated = self._calculate_reward()

        self.episode_len_buffer += 1
        truncated = self.episode_len_buffer > 2000

        dones = torch.bitwise_or(terminated, truncated)

        # Before reset, get the true final observations for truncated environments
        final_obs = self._get_obs().clone()

        self.action_t_2 = self.action_t_1.clone()
        self.action_t_1 = self.action_t.clone()
        self.previous_joint_velocity = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)

        reset_ids = torch.nonzero(dones).flatten()
        if len(reset_ids) > 0:
            self.reset_env_idx(reset_ids)

        self.timestep += 1
        observation = self._get_obs()
        
        info = {
            "final_obs": final_obs.detach().cpu().numpy(),
            "terminated": terminated.detach().cpu().numpy(),
            "truncated": truncated.detach().cpu().numpy()
        }
        return observation.detach().cpu().numpy(), reward.detach().cpu().numpy(), dones.detach().cpu().numpy(), info
    

if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level="warning")
    env = WalkENV(render=True, n_batch=1)
    
    state, info = env.reset()

    while True:
        action = env.action_space.sample()
        state, reward, dones, info = env.step(action)

