import numpy as np
import random

class QvalueCMAC:
    """
    This implements the TD(lambda) algorithm, using CMAC tiling as a linear function approximation.
    Note that the implementation is stateless - the agent needs to store and provide current state, action etc. when it is needed.
    Based on code written by Sridhar Mahadevan and Richard Sutton, with few modifications.
    Code can be found here - http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar.html (Mahadevan's earlier C++ version w/ X11)
    Additional details can be found in Sutton & Barto.
    """

    def __init__(self,features_properties_dict,num_tilings,action_mnemonics_list,Lambda=0.9,Alpha=0.5,Q0=0):
        """Constructs the QvalueCMAC object

        Attributes:
            features_properties_dict: dictionary containing the features, and per-feature properties as a tuple. The tuple is (min feature range, max feature range, number of bins for feature).
            num_tulings: number of required tilings.
            action_mnemonics_list: list of action mnemonics.
            Lambda: discount rate (for eligibility traces).
            Alpha: learning rate.
            Q0: init value for the weights vector.

        Example:
            qvalue_cmac = QvalueCMAC({'pos':(-1.2,0.6,8),'vel':(-0.07,0.07,9)},10,["neutral","forward","reverse"],Q0=1)
        """


        self.cmac = CMAC(features_properties_dict,num_tilings)
        self.Lambda = Lambda #eligiability discount rate
        self.Alpha = Alpha #learning rate


        #translate the action mnemonics into integer, to be using internally.
        self.actions_map = dict()
        for idx,val in enumerate(action_mnemonics_list):
            self.actions_map[val] = idx

        #generate a canonical state ordering, to be used internally in all functions
        self.state_order = features_properties_dict.keys()

        #generate the dimensions for the weight and eligibility matrices.
        self.dimensions = ()
        for d in self.state_order:
            self.dimensions += (self.cmac.num_bins[d]+1,)
        self.dimensions += (num_tilings,len(self.actions_map))

        self.weight = np.ones(self.dimensions) * Q0/float(self.cmac.num_tilings)
        self.reset_eligibility()

    def _get_state_as_tile_tuple(self,state_dict,tile):
        #translate state to tile, and return return a canonical representation as a tuple

        tile_dict = self.cmac.get_tiles(state_dict,tile)
        t = ()
        for key in self.state_order:
            t += (tile_dict[key][tile],)

        return t

    def get_qvalue(self,state_dict,action_mnemonics):
        """Returns the qvalue of state and action pair.

        Attributes:
            state_dict: a dictionary maps between a feature and its state value.
            action_mnemonics: the mnemonics of the action to be evaluated.

        Example:
            qvalue = qvalue_cmac.get_qvalue({'vel':0.1,'pos',0.5},"forward")

        """
        value = 0
        #from action name to action enumeration
        action_val = self.actions_map[action_mnemonics]
        for tile in range(self.cmac.num_tilings):
            #generate key: weight[tiles['pos][tile],tiles['vel'][tile],tile,action]
            t = self._get_state_as_tile_tuple(state_dict,tile)
            t += (tile,action_val)
            value += self.weight[t]

        return value



    def _get_best_action_and_qvalue_for_state(self,state_dict):
        #pick any random action to start with, just so we are not biased when there is a tie
        best_action = random.choice(self.actions_map.keys())
        best_qvalue = self.get_qvalue(state_dict,best_action)

        for action in self.actions_map:
            qvalue = self.get_qvalue(state_dict,action)
            if (qvalue > best_qvalue):
                best_qvalue  = qvalue
                best_action  = action

        return [best_action,best_qvalue]

    def get_best_action_for_state(self,state_dict):
        """Returns the mnemonics of the action with the highest qvalue.

        Attributes:
            state_dict: a dictionary maps between a feature and its state value.

        Example:
            action_mnemonics = qvalue_cmac.get_best_action_for_state({'vel':0.1,'pos',0.5})

        """
        return self._get_best_action_and_qvalue_for_state(state_dict)[0]

    def get_best_qvalue_for_state(self,state_dict):
        """Returns the max qvalue of all actions in a given state.
        Attributes:
            state_dict: a dictionary maps between a feature and its state value.

        Example:
            qvalue = qvalue_cmac.get_best_qvalue_for_state({'vel':0.1,'pos',0.5})

        """
        return self._get_best_action_and_qvalue_for_state(state_dict)[1]


    def update_eligibilities(self,state_dict,action):
        """Updates the eligibility trace - decay it according to Lamda, and set entries of state and action to 1.

        Attributes:
            state_dict: a dictionary maps between a feature and its state value.
            action: the action mnemonics

        Example:
            qvalue_cmac.update_eligibilities({'vel':0.1,'pos',0.5},"forward")

        """

        #decay all eligibilities by lambda
        self.eligibility *= self.Lambda
        #mnemonics to value
        act = self.actions_map[action]
        for tile in range(self.cmac.num_tilings):
            t = self._get_state_as_tile_tuple(state_dict,tile)
            self.eligibility[t+(tile,slice(None))] = 0
            self.eligibility[t+(tile,act)] = 1

        """
        after this update:
        - all traces have been decayed
        - trace of current state, current action (all tiles) is 1
        - traces of current state, any action != current action (all tiles) is 0
        """


    def reset_eligibility(self):
        """Resets the eligibility traces to 0.
        """
        self.eligibility = np.zeros(self.dimensions)


    def update_weights(self,r,old_qval,new_qval):
        """Updates the weights according to r, old value of q function, and new value of q function.

         Attributes:
            r: reward
            old_qval: previous q value of the q function
            new_qval: current q value of the q function

        Example:
            qvalue_cmac.update_weights(-1, 0.78,0.82)
        """
        delta = (self.Alpha/float(self.cmac.num_tilings))*(r + new_qval - old_qval) * self.eligibility
        self.weight += delta


class CMAC:
    """
    CMAC tiling class
    Based on code written by Sridhar Mahadevan and Rich Sutton.
    Code can be found here - http://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar.html (Mahadevan's earlier C++ version w/ X11)
    Additional details can be found in Rich's book, some intuition can be found here - https://groups.google.com/forum/#!searchin/rl-list/CMAC$20and$20Hashing%7Csort:relevance/rl-list/c2nLzWJ91vY/MLhNGbkwCpcJ
    Wikipedia: https://en.wikipedia.org/wiki/Cerebellar_model_articulation_controller
    """

    def __init__(self,features_properties_dict,num_tilings):
        """Constructs the CMAC object

        Attributes:
            features_properties_dict: dictionary containing the features, and per-feature properties as a tuple. The tuple is (min feature range, max feature range, number of bins for feature).
            num_tulings: number of required tilings.

        Example:
            cmac = CMAC({'pos':(-1.2,0.6,8),'vel':(-0.07,0.07,9)},10)
        """

        self.min_range=dict()
        self.max_range=dict()
        self.interval_size=dict()
        self.num_bins=dict()

        for key in features_properties_dict:
            self.min_range[key],self.max_range[key],self.num_bins[key] = features_properties_dict[key]

            """
            Calculate the interval size for each feature - the size of each bin. If the feature range is 2.3,
            and I have 13 bins, each bin will be 2.3/13 =  0.176.
            """
            self.interval_size[key] = (self.max_range[key] - self.min_range[key])/float(self.num_bins[key])

        self.num_tilings = num_tilings
        self.offset=dict()

        """
        Calculate an offset per feature, per tiling.
        The offset is just a random number between 0 and interval_size. Later on, it will be used on the tiling retrieval.
        """

        for key in features_properties_dict:
            self.offset[key]=dict()
            for tiling in range(num_tilings):
                #self.offset[key][tiling] = 0.5 * self.interval_size[key]#
                self.offset[key][tiling] = np.random.random() * self.interval_size[key]



    def get_tiles(self,state_dict,tile=None):
        """Get a dictionary of state representation per tiling. This is the heart of the CMAC tiling method.
        The function returns a 2d hash, which keys are [feature][tile] and value is the translation of the state to the tiling domain.

        Attributes:
            state_dict: a dictionary maps between a feature and its state value.
            tile (optional): will return a dictionary containing only the specified tile (to save time when only a single tile is required). If not provided, will return a dictionary that contains all tiles.

        Example:
            cmac = get_dict({'pos':-1.1,'vel':-0.07},tile=7)
        """

        tiles_dict=dict()
        assert(sorted(state_dict.keys()) == sorted(self.min_range.keys()))
        tiles_list = range(self.num_tilings) if tile == None else [tile]

        for key in state_dict:
            tiles_dict[key]=dict()
            for tile in tiles_list:
                tiles_dict[key][tile] = int( float(state_dict[key] - self.min_range[key] - self.offset[key][tile])/float(self.interval_size[key]))

        return tiles_dict

