class RL_Config(object):
    def __init__(self, args) -> None:
        super().__init__()
        
        if args.debug:
            ###################################################
            # Debug Setting 
            ###################################################
            self.OBSERVE = 10 # timesteps to observe before training
            self.REPLAY_MEMORY = 500 # number of previous transitions to remember
            self.BATCH_SIZE = 2 # size of minibatch
            self.GAMMA = 0.98 # decay rate of past observations
            self.UPDATE_TIME = 10
            self.RANDOM_ACTION = 20
        else:
            ###################################################
            # Training Setting 
            ###################################################
            # RL model and training
            self.OBSERVE = 50 # timesteps to observe before training
            self.REPLAY_MEMORY = 500 # number of previous transitions to remember
            self.BATCH_SIZE = 8 # size of minibatch
            self.GAMMA = 0.98 # decay rate of past observations
            self.UPDATE_TIME = 20
            if args.resume:
                self.RANDOM_ACTION = 0
            else:
                self.RANDOM_ACTION = 50


