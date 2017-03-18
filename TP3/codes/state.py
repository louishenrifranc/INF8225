class State:
    def __init__(self, layer_depth):
        self.layer_depth = layer_depth


class ConvolutionState(State, object):
    def __init__(self, *args):
        """
        State representing a convolution state
        :param args: List (layer_depth, kernel_size, stride=1, nb_channel, representation_bin)
               A list representing uniquely a Convolution layer
        """
        super(self.__class__, self).__init__(args[0])
        self.receptieve_field_size = args[1]
        self.stride = args[2]
        self.receptive_fields = args[3]
        self.representation_size = args[4]

    def __repr__(self):
        return "{}: C({}, {}, {})".format(self.layer_depth,
                                          self.receptive_fields,
                                          self.receptieve_field_size,
                                          self.stride,
                                          )

    def __eq__(self, other):
        if not isinstance(other, ConvolutionState):
            return False
        return self.layer_depth == other.layer_depth \
               and self.receptieve_field_size == other.receptieve_field_size \
               and self.stride == other.stride \
               and self.receptive_fields == other.receptive_fields \
               and self.representation_size == other.representation_size

    def __hash__(self):
        return hash((self.layer_depth, self.receptieve_field_size, self.stride, self.receptive_fields,
                     self.representation_size))


class PoolingState(State, object):
    def __init__(self, *args):
        """
        State representing a Pooling State
        :param args: List (layer_depth, tuple(kernel_size, stride), representation_bin
            A list representing uniquely a Pooling layer
        """
        super(self.__class__, self).__init__(args[0])
        self.pair_receptive_field_size_strides = args[1]
        self.representation_size = args[2]

    def __repr__(self):
        return "{}:P({}, {})".format(self.layer_depth, *self.pair_receptive_field_size_strides)

    def __eq__(self, other):
        if not isinstance(other, PoolingState):
            return False
        return self.layer_depth == other.layer_depth \
               and self.pair_receptive_field_size_strides[0] == other.pair_receptive_field_size_strides[0] \
               and self.pair_receptive_field_size_strides[1] == other.pair_receptive_field_size_strides[1] \
               and self.representation_size == other.representation_size

    def __hash__(self):
        return hash((self.layer_depth, self.pair_receptive_field_size_strides, self.representation_size))


class FullyConnectedState(State, object):
    def __init__(self, *args):
        """
        State corresponding to a fully connected layer
        :param self:
        :param args: List (layer_depth, nb_previous_fc_layers, nb_hidden)
            A list representing uniquely a FC layer
        """
        super(self.__class__, self).__init__(args[0])
        self.consecutive_fc_layers = args[1]
        self.nb_hidden = args[2]

    def __repr__(self):
        return "{}:FC({})".format(self.layer_depth, self.nb_hidden)

    def __eq__(self, other):
        if not isinstance(other, FullyConnectedState):
            return False
        return self.layer_depth == other.layer_depth \
               and self.consecutive_fc_layers == other.consecutive_fc_layers \
               and self.nb_hidden == other.nb_hidden

    def __hash__(self):
        return hash((self.layer_depth, self.consecutive_fc_layers, self.nb_hidden))


class TerminateState(State, object):
    def __init__(self, prev_state):
        """
        State corresponding to a final state (max global pooling)
        :param prev_state: Previous state (FullyConnectedState, PoolingState, ConvolutionState)
        """
        super(self.__class__, self).__init__(prev_state.layer_depth + 1)
        self.previous_state = prev_state

    def __repr__(self):
        return "{}:T()".format(self.layer_depth)

    def __eq__(self, other):
        if not isinstance(other, TerminateState):
            return False
        return self.layer_depth == other.layer_depth \
               and self.previous_state == other.previous_state

    def __hash__(self):
        return hash((self.layer_depth, self.previous_state))
