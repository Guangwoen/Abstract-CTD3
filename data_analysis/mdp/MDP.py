class Node:
    def __init__(self, state):
        self.state = state
        self.edges = set()

    def add_edge(self, next_node, action, reward, done, cost, weight, prob):
        self.edges.add(Edge(next_node, action, reward, done, cost, weight, prob))

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def __hash__(self):
        return hash(id(self.state))

    def to_dict(self):
        return {
            'state': self.state,
            'edges': [edge.to_dict() for edge in self.edges]
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data['state'])
        node.edges = {Edge.from_dict(edge_data) for edge_data in data.get('edges', [])}
        return node


class Edge:
    def __init__(self, next_node, action, reward, done, cost, weight, prob):
        self.next_node = next_node
        self.action = action
        self.reward = reward
        self.done = done
        self.cost = cost
        self.weight = weight
        self.prob = prob

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.next_node == other.next_node
        return False

    def __hash__(self):
        return hash((
            hash(id(self.next_node)),
            tuple(self.action),
            tuple(self.reward),
            self.done,
            tuple(self.cost),
            self.weight,
            self.prob
        ))

    def to_dict(self):
        return {
            'next_node': self.next_node.to_dict(),
            'action': self.action,
            'reward': self.reward,
            'done': self.done,
            'cost': self.cost,
            'weight': self.weight,
            'prob': self.prob
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            next_node=Node.from_dict(data['next_node']),
            action=data['action'],
            reward=data['reward'],
            done=data['done'],
            cost=data['cost'],
            weight=data['weight'],
            prob=data['prob']
        )


class Attr:
    def __init__(self, state):
        self.state = state
        self.probs = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.done = []
        self.weights = []
        self.next_states = []
        self.max_action = -100
        self.min_action = 100
        for edge in state.edges:
            self.probs.append(edge.prob)
            self.actions.append(edge.action)
            self.done.append(edge.done)
            self.weights.append(edge.weight)
            self.rewards.append(edge.reward)
            self.next_states.append(edge.next_node)
            self.costs.append(edge.cost)
            # TODO 最大最小需要根据动作维度确定，这里先用一维
            self.min_action = min(self.min_action, edge.action[0])
            self.max_action = max(self.max_action, edge.action[0])

    def to_dict(self):
        return {
            'state': self.state.to_dict(),
            'probs': self.probs,
            'actions': self.actions,
            'rewards': self.rewards,
            'costs': self.costs,
            'done': self.done,
            'weights': self.weights,
            'next_states': [state.to_dict() for state in self.next_states],
            'max_action': self.max_action,
            'min_action': self.min_action
        }

    @classmethod
    def from_dict(cls, data):
        attr = cls(data['state'])
        attr.probs = data['probs']
        attr.actions = data['actions']
        attr.rewards = data['rewards']
        attr.costs = data['costs']
        attr.done = data['done']
        attr.weights = data['weights']
        attr.next_states = [Node.from_dict(state_data) for state_data in data['next_states']]
        attr.max_action = data['max_action']
        attr.min_action = data['min_action']
        return attr