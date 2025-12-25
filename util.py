import yaml
import numpy as np 


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        return self.optimizer.state_dict()
        
    def load_state_dict(self,state):
        self.optimizer.load_state_dict(state)

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2
        self.delta = min(1024,self.delta)

    def update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr