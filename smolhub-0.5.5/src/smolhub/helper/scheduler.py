
import math

class CustomLRScheduler:
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr, _type):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.it = 0
        self._last_lr = [max_lr]  # Initialize with max_lr (matching PyTorch convention)
        self.type = _type

    def step(self):
        
        self._last_lr = [self._get_lr()]  # Store as list
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._last_lr[0]
        self.it += 1

    def get_last_lr(self):
        return self._last_lr  # Returns list to match PyTorch convention
    
    def _get_lr(self):

        if(self.type == 'cycle'):
            cycle = math.floor(1 + self.it / (2 * self.warmup_iters))
            x = abs(self.it / self.warmup_iters - 2 * cycle + 1)
            return self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - x))

        elif(self.type == 'cosine'):
            # 1) linear warmup for warmup_iters steps
            if self.it < self.warmup_iters:
                return self.max_lr * (self.it + 1) / (self.warmup_iters + 1)
            # 2) if it > lr_decay_iters, return min learning rate
            if self.it > self.lr_decay_iters:
                return self.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (self.it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
            return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'lr_decay_iters': self.lr_decay_iters,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'it': self.it
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.lr_decay_iters = state_dict['lr_decay_iters']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.it = state_dict['it']
