import torch.nn as nn
import torch

class NTXentLoss(nn.Module):
    def __init__(self,temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self,z1,z2):
        """
        z1, z2: tensor of shape (batch_size=N, out_dim=128)
        z1[i] and z2[i] are a positive pair
        """
        batch_size = z1.shape[0]
        z = torch.cat([z1,z2], dim = 0) # stack the two views together ( 2N * out_dim)

        sim = torch.matmul(z, z.T) / self.temperature # shape [ 2N * 2N]

        # create a mask
        mask = torch.eye(2 * batch_size, dtype = torch.bool, device = z.device)
        #The diagonal corresponds to each sample compared with itself.
        sim.masked_fill_(mask, float('-inf'))
        # Now the diagonal elements are -inf, so they won't contribute to the softmax.
        #Because every sample is maximally similar to itself, and we do not want the model to cheat by picking itself.

        positives = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim = 0)

        # positive for row i in first half is at column i + N
        # positive for row i in second half is at column i - N  

        denominator = torch.logsumexp(sim, dim = 1) # log( sum over all non-self samples exp(sim[i, k]) )

        loss = -positives + denominator #NTXent loss: -log(exp(sim_pos)) + log( sum exp(sim_neg) )
        
        return loss.mean()



        




