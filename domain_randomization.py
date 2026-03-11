import torch

class DomainRandomization:
    def __init__(self, robot, n_envs, joints_local_idx):
        self.robot = robot
        self.n_envs = n_envs
        self.joints_local_idx = joints_local_idx
        
    def randomize(self, env_idx = None):
        randomize_envs = torch.arange(self.n_envs)
        if env_idx == None:
            randomize_envs = self.n_envs

        total_envs = len(randomize_envs)

        self.robot.set_friction_ratio(
            friction_ratio=0.5 + torch.rand(total_envs, self.robot.n_links),
            links_idx_local=range(self.robot.n_links),
            envs_idx = randomize_envs
        )

        self.robot.set_mass_shift(
            mass_shift=torch.rand(total_envs, self.robot.n_links) * 0.5,
            links_idx_local=range(self.robot.n_links),
            envs_idx = randomize_envs
        )

        self.robot.set_COM_shift(
            com_shift=-0.05 + 0.1 * torch.rand(total_envs, self.robot.n_links, 3),
            links_idx_local=range(self.robot.n_links),
            envs_idx = randomize_envs
        )
