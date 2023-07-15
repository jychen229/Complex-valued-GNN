
import torch

class AngularAggLayer(nn.Module):
    def __init__(self, num_nodes, num_class, input_dim, labels, dropout):
        super(AngularAggLayer, self).__init__()
        self.register_buffer("mean_tensor", torch.randn(num_nodes, num_class))
        self.register_buffer("mag", torch.randn(1, input_dim))
        self.k = num_class
        self.N = num_nodes
        self.labels = labels
        self.dropout = dropout
        num = int(num_class*(num_class-1)/2)
        self.theta = nn.Parameter(torch.zeros(num,))
        
    def pre_norm(self, features):
        max_values, _ = torch.max(features, dim=0)
        theta = torch.arccos(features/max_values)
        new_fea = max_values * torch.cos(theta) + 1j*features*torch.sin(theta)
        return new_fea, max_values
    
    def post_norm(self, features):
        feature_norm = torch.abs(features)
        unit_norm_result = features / feature_norm
        normalized_result = unit_norm_result * self.mag
        return normalized_result
    
    # ! maybe trace the grad 
    def negative_symmetric_theta(self):
        matrix = torch.zeros(self.k, self.k)
        triu_indices = torch.triu_indices(self.k, self.k, offset=1)  # 上三角
        tril_indices = torch.tril_indices(self.k, self.k, offset=-1)  # 下三角
        matrix[triu_indices[0], triu_indices[1]] = self.theta
        matrix[tril_indices[1], tril_indices[0]] = -self.theta
        return matrix
    
    def get_centers(self, features, labels):
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.k)
        complex_one_hot = one_hot.t().to(torch.complex64)
        sum_by_label = torch.matmul(complex_one_hot, features)
        count_by_label = one_hot.sum(dim=0).unsqueeze(1)
        mean_tensor = sum_by_label / count_by_label
        return mean_tensor
    
    #! hard attention
    def get_label(self, feature, p = 1):
        dist = complex_dist(feature, self.mean_tensor, p = p)
        closest = torch.argmin(dist, dim=1)
        return closest
    
    def generate_matrix(self, w, label):
        label_indices = label.long()
        A_hat = w[label_indices.unsqueeze(1), label_indices.unsqueeze(0)]
        return A_hat
    
    def forward(self, x, A, l0=False):
        if l0:
            norm_features, self.mag = self.pre_norm(x)
        else:
            norm_features = x
            
        raw = norm_features
        self.mean_tensor = self.get_centers(norm_features, self.labels)
        
        #self.restr_theta()
        matrix = self.negative_symmetric_theta()
        
        fake_label = self.get_label(norm_features)
        A_hat = self.generate_matrix(matrix, fake_label)
        A = torch.where(A > 0, torch.tensor(1.0), A)
        A = torch.mul(A, A_hat)
        adj = torch.cos(A)+1j*torch.sin(A)
        message = torch.matmul(adj, norm_features)
        return self.post_norm(raw+message)
        

def complex_dist(c1,c2,p=1):
    real_part1 = c1.real
    imaginary_part1 = c1.imag
    real_part2 = c2.real
    imaginary_part2 = c2.imag
    
    real_distances = torch.cdist(real_part1, real_part2, p=p)
    imaginary_distances = torch.cdist(imaginary_part1, imaginary_part2, p =p)
    complex_distances = torch.sqrt(real_distances**2 + imaginary_distances**2)

    return complex_distances


class CGNN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, labels, num_nodes):
        super(CGNN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(AngularAggLayer(num_nodes, nclass, nhidden, labels, dropout))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        
    def complex_relu(self, input):
        return F.relu(input.real).type(torch.complex64)+1j*F.relu(input.imag).type(torch.complex64)

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            if i ==0:
            #layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                layer_inner = self.complex_relu(con(layer_inner,adj,True))
            else:
                layer_inner = self.complex_relu(con(layer_inner,adj))

        #layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        #--------#
        layer_inner = torch.angle(layer_inner)                   
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
        

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout,self).__init__()
        self.p = p
        
    def complex_dropout(self, input, p=0.5, training=True):
        # need to have the same dropout mask for real and imaginary part, 
        device = input.device
        mask = torch.ones(input.real.shape, dtype = torch.float32, device = device)
        mask = F.dropout(mask, p, training)*1/(1-p)
        mask.type(input.real.dtype)
        real,imag = mask * input.real,mask * input.imag
        return torch.complex(real,imag)

    def forward(self,input):
        if self.training:
            return self.complex_dropout(input,self.p)
        else:
            return input


#---------- To Do --------------

#complex distance  
def get_label_pro(feature, mean_tensor):
    '''
    GAT pattern 
    '''
    N = feature.shape[0]
    d = feature.shape[1] 
    k = mean_tensor.shape[0]
    
    fea_repeated_in_chunks = feature.repeat_interleave(N, dim=0)
    fea_repeated_alternating = feature.repeat(N, 1)
    all_combinations_matrix = torch.cat([fea_repeated_in_chunks, fea_repeated_alternating], dim=1)
    feature = all_combinations_matrix.view(N, N, 2*d)
    
    first_part = feature[:, :, :d]
    second_part = feature[:, :, d:]

    dist_first = complex_dist(first_part.reshape(N*N, d),mean_tensor.reshape(k, d))
    dist_second = complex_dist(second_part.reshape(N*N, d),mean_tensor.reshape(k, d))

    # 获取前d维和后d维最接近的两个类别索引
    closest_first = torch.argmin(dist_first, dim=1)
    closest_second = torch.argmin(dist_second, dim=1)

    return closest_first.view(N,N), closest_second.view(N,N)




#loss coa m *theta  -n

def Augular_loss():
    pass
