

class Config:
    
    class Training:
        training_timesteps = 1000
        simulation_timesteps = 100
        timestep = 0.1
        worlds_parallised = 2
        seed = 1
        periodic_save_interval = 50

    class Mappo:
        ppo_epochs = 4
        gamma = 0.99
        gae_lambda = 0.95
        clip_coef = 0.2
        vf_coef = 0.5
        ent_coef = 0.01
        actor_learning_rate = 0.001
        critic_learning_rate = 0.001

    class Actor:
        lstm_hidden_size = 64
        feature_dim = 128

    class Comms:
        vocab_size = 8
        communication_size = 16
        num_comms = 2
        rq_levels = 3

        class CommunicationType:
            value = "DISCRETE"

            def upper(self):
                return self.value.upper()

        communication_type = CommunicationType()

    class AimTraining:
        aim_learning_rate = 0.001
        obs_runs = 5
        obs_runs_noise = 0.1
        aim_batch_size = 16

    training = Training()
    mappo = Mappo()
    actor = Actor()
    comms = Comms()
    aim_training = AimTraining()

    def save(self, path):
        pass