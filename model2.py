import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import six


MAX_NUMBER_ATOM = 140
NUM_EDGE_TYPE = 4
K = 10
class MLP(chainer.Chain):
    def __init__(self, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        with self.init_scope():
            self.l1 = L.Linear(self.hid_dim)
            self.l2 = L.Linear(self.hid_dim)
            self.l3 = L.Linear(self.out_dim)

    def __call__(self, x, y):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        self.loss = F.sigmoid_cross_entropy(h, y)
        self.accuracy = F.binary_accuracy(h, y)
        return self.loss, self.accuracy

class VarNet(chainer.Chain):
    def __init__(self, hid_dim, out_dim, p = 0.01, train = True):
        super(VarNet, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.p = p
        self.train = train
        with self.init_scope():
            self.l1 = L.Linear(self.hid_dim)
            self.l2 = L.Linear(self.hid_dim)
            self.l3 = L.Linear(self.out_dim)


    def logistic_func(self, x):
        return 1/(1 + F.exp(-K * x))

    def __call__(self, fp, y):
        mean_activation = F.mean(fp, axis=0)
        rho = 0.01
        zero_array = chainer.Variable(numpy.zeros(mean_activation.shape, dtype=numpy.float32))
        small_array = zero_array + 0.001

        cond = (mean_activation.data != 0)
        cond = chainer.Variable(cond)
        mean_activation = F.where(cond, mean_activation, small_array)

        self.kl_div = rho * F.sum(F.where(cond, self.p * F.log(self.p/mean_activation) + (1 - self.p) * F.log((1 - self.p) / (1 - mean_activation)), zero_array))
        # sampling z
        eps = numpy.random.uniform(0.0, 1.0, fp.data.shape).astype(numpy.float32)
        eps = chainer.Variable(eps)


        if self.train == True:
            z = self.logistic_func(fp - eps)
            #z = fp
        else:
            z = fp
        h = F.relu(self.l1(z))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        self.rec_loss = F.sigmoid_cross_entropy(h, y)
        self.accuracy = F.binary_accuracy(h, y)
        self.loss = self.rec_loss + self.kl_div
        return self.loss, self.accuracy
    def predict(self, fp):
        z = fp
        h = F.relu(self.l1(z))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return F.sigmoid(h)



class SubNFP(chainer.Chain):

    def __init__(self, hidden_dim, out_dim, max_degree):
        super(SubNFP, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.gate_weight = L.Linear(out_dim, 1)
            self.hidden_weights = chainer.ChainList(
                *[L.Linear(hidden_dim, hidden_dim)
                  for _ in range(num_degree_type)]
            )
            self.output_weight = L.Linear(hidden_dim, out_dim)
            self.edge_layer = L.Linear(hidden_dim, NUM_EDGE_TYPE * hidden_dim)
        self.max_degree = num_degree_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
    def __call__(self, x, h, adj, deg_conds, counts):
        s0, s1, s2 = x.shape
        m = F.reshape(self.edge_layer(F.reshape(x, (s0 * s1, s2))), (s0, s1, s2, NUM_EDGE_TYPE))
        m = F.transpose(m, (0, 3, 1, 2))
        adj = F.reshape(adj, (s0 * NUM_EDGE_TYPE, s1, s1))
        m = F.reshape(m, (s0 * NUM_EDGE_TYPE, s1, s2))
        m = F.batch_matmul(adj, m)
        m = F.reshape(m, (s0, NUM_EDGE_TYPE, s1, s2))
        m = F.sum(m, axis=1)
        m = F.sigmoid(m)

        s0, s1, s2 = m.shape
        zero_array = numpy.zeros(m.shape, dtype=numpy.float32)
        ms = [F.reshape(F.where(cond, m, zero_array), (s0 * s1, s2)) for cond in deg_conds]
        out_x = 0
        for hidden_weight, m in zip(self.hidden_weights, ms):
            out_x = out_x + hidden_weight(m)
        out_x = F.sigmoid(out_x)


        incorrect_part = numpy.zeros(out_x.shape, dtype=numpy.float32)
        for s_index in range(s0):
            out_x.data[counts[s_index] + s_index * s1:(s_index + 1) * s1, :] = 0.0


        dh = self.output_weight(out_x)
        dh = F.sigmoid(dh)
        #dh = out_x
        #incorrect_part = numpy.zeros(dh.shape, dtype=numpy.float32)
        for s_index in range(s0):
            dh.data[counts[s_index] + s_index * s1:(s_index + 1) * s1, :] = 0.0


        #gate = F.sigmoid(self.gate_weight(dh))
        # dh = dh * gate
        #gate = F.tile(gate, self.out_dim)
        #dh = dh * gate


        dh = F.sum(F.reshape(dh, (s0, s1, self.out_dim)), axis=1)
        #print(dh.data)
        out_x = F.reshape(out_x, (s0, s1, s2))
        out_h = h + dh
        return out_x, out_h


class NFP(chainer.Chain):
    def __init__(self, hidden_dim, out_dim, max_degree, n_atom_types, radius):
        super(NFP, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.embed = L.EmbedID(n_atom_types, hidden_dim)
            self.layers = chainer.ChainList(
                *[SubNFP(hidden_dim, out_dim, max_degree)
                  for _ in range(radius)])
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_degree_type = num_degree_type
        self.radius = radius
        self.out_dim = out_dim


    def __call__(self, adj, atom_array):

        counts = []
        for list_atom in atom_array:
            list_atom = numpy.array(list_atom)
            count = numpy.count_nonzero(list_atom)
            counts.append(count)
        x = self.embed(atom_array)
        h = 0
        degree_mat = F.sum(adj, axis=1)
        degree_mat = F.sum(degree_mat, axis=1)
        # print("[DEBUG]:", degree_mat.shape)
        # print("xshape:", x.shape)
        deg_conds = [self.xp.broadcast_to(((degree_mat - degree).data == 0)[:, :, None], x.shape)
                     for degree in range(1, self.num_degree_type + 1)]
        h_list = []
        #print("number of layers:", self.radius)
        for l in self.layers:
            x, h = l(x, h, adj, deg_conds, counts)
        s0, s1 = h.data.shape
        #print(h.data)

        counts = F.reshape(chainer.Variable(numpy.array(counts, dtype=numpy.float32)), (len(counts), 1))
        #print(F.tile(counts, (1, s1)) * self.radius)
        h = h / (F.tile(counts, (1, s1)) * self.radius)
        # h = F.sigmoid(h)
        # h = F.softmax(h)

        return h

class Predictor(chainer.Chain):
    def __init__(self, nfp, hid_dim, out_dim):
        super(Predictor, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        with self.init_scope():
            self.nfp = nfp
            self.mlp = MLP(self.hid_dim, self.out_dim)

    def __call__(self, adj, atom_types, label):
        x = self.nfp(adj, atom_types)
        self.loss, self.accuracy = self.mlp(x, label)

        return self.loss

    def predict(self, adj, atom_types):
        x = self.nfp(adj, atom_types)
        y = self.mlp.predict(x)
        return y

class VPredictor(chainer.Chain):
    def __init__(self, nfp, hid_dim, out_dim, train = True):
        super(VPredictor, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.train = train
        with self.init_scope():
            self.nfp = nfp
            self.vmlp = VarNet(self.hid_dim, self.out_dim, 0.001)
    def setMode(self, mode):
        self.train = mode
        self.vmlp.train = mode
    def __call__(self, adj, atom_types, y):
        fps = self.nfp(adj, atom_types)
        self.loss, self.accuracy = self.vmlp(fps, y)
        self.rec_loss = self.vmlp.rec_loss
        self.kl_dv = self.vmlp.kl_div
        return self.loss
    def predict(self, adj, atom_types):
        x = self.nfp(adj, atom_types)
        y = self.vmlp.predict(x)
        return y

