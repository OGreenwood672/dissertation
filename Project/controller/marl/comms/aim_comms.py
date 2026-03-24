# import torch
# import torch.nn as nn

# from controller.marl.models.aim import AIM
# from controller.marl.models.encoders.encoder import Encoder


# from .comms import Comms



# class AimComms(Comms):

#     def __init__(self, config, device):

#         folder = AIM.get_folder(config)
#         encoder = Encoder.load(folder)

#         aim_codebook = torch.tensor(encoder.get_codebooks(), dtype=torch.float32, device=device)
#         self.register_buffer('aim_codebook', aim_codebook)

#         self.hq_levels = aim_codebook.shape[0]

#         # Teaching the fixed codebook
#         # Sender must understand what its sending
#         # Predict other agent rewards
#         self.predict_agents_returns = nn.Linear(comm_dim * num_comms + lstm_hidden_size, num_agents)
        
#         # Reciever must understand how it is getting effected
#         # Predict critic value
#         self.predict_critic_value = nn.Linear(comm_dim * num_comms + lstm_hidden_size, 1)

#         self.comm_heads = nn.ModuleList([
#             nn.Linear(lstm_hidden_size + comm_dim * hq_level * num_comms, num_comms * vocab_size)
#             for hq_level in range(self.hq_levels)
#         ])

#         # Intent Predictors
#         # self.predict_intent_sender = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)
#         # self.predict_intent_receiver = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)

#         self.predict_intent_sender_goal = nn.Linear(comm_dim * num_comms + lstm_hidden_size, 2)
#         self.predict_intent_receiver_goal = nn.Linear(comm_dim * num_comms + lstm_hidden_size, 2)
