import torch
import numpy as np
import gpytorch.utils.lanczos as lanczos
import math

def get_sample(matrix_parameters):
    # Instantiate test set
    N,d = matrix_parameters["N"],matrix_parameters["d"]
    if "operation" in matrix_parameters:
        op = matrix_parameters["operation"]
        M = RandomMatrixDataSet(N, d, op)
    else: 
        M = RandomMatrixDataSet(N, d)
    # If the condition number is specified
    if "cond" in matrix_parameters:
        M.from_condition_number(matrix_parameters["cond"])
    # If the eigenvalues are specified
    elif "eigenvalues" in matrix_parameters:
        M.from_eigenvalues(eigenvalues=matrix_parameters["eigenvalues"], diagonal=matrix_parameters["diagonal"], similar = matrix_parameters["similar"])
    # If mu & sigma specified, eigenvalues are drawn from IID normal distributions  N(mu,sigma^2)
    elif "mu" and "sigma" in matrix_parameters:
        mu, sigma = matrix_parameters["mu"], matrix_parameters["sigma"]
        M.from_eigenvalues(mu=mu, sigma=mu, similar=matrix_parameters["similar"], diagonal=matrix_parameters["diagonal"])
    # Otherwise just sample a matrix with standard normal distributed elements
    elif "dist" in matrix_parameters:
        M.from_dist(matrix_parameters["dist"])
    else:
        M.from_rand() #M.from_randn() #fix so one can choose between these
        if "symmetric" in matrix_parameters: # Construct symmetric random matrix
            if matrix_parameters["symmetric"]: #Create Wigner matrix
                M.X = torch.triu(M.X,0) + torch.transpose(torch.triu(M.X,1),2,3)  #torch.matmul(M.X,torch.transpose(M.X,2,3)) 
        if "lanczos" in matrix_parameters:
            if matrix_parameters["lanczos"]:
                M.get_lanczos_tridiag()
        
    #Flatten and append determinant as a feature
    #Maybe this is not very smart, since training/testing will be restricted to this matrix type. 
    if "det" in matrix_parameters and matrix_parameters["det"]: #This will throw an error if "det" is not specified, maybe change 
        M.compute_determinant()
        m = torch.flatten(M.X, start_dim = 2)
        M.X_with_det = torch.cat((m,M.det[:,None,:]),2)
        
    elif "det_channel" in matrix_parameters and matrix_parameters["det_channel"]:
        M.compute_determinant()
        m = torch.eye(d)*M.det[:, None] #Try both det and 1/det!
        m = m[:,None,:,:]
        M.X_with_det = torch.cat((M.X,m),1)
    elif "permutations" in matrix_parameters:       
            M.permute(matrix_parameters["permutations"])                  
    return M

class RandomMatrixDataSet:
    def __init__(self, N, d = 3, operation=torch.linalg.inv):
        self.N = N
        self.d = d
        self.X = None
        self.Y = None
        self.operation = operation
        self.cond = None
        self.det = None
        self.X_with_det = None
        self.X_with_permutations = None
        self.X_Hess = None
        
    def from_rand(self, r1 = -10, r2 = 10): 
        self.X = (r1-r2)*torch.rand(self.N,1,self.d,self.d) + r2
    
    def from_randn(self):
        self.X = torch.randn(self.N,1,self.d,self.d)
    
    def from_dist(self, dist):
        self.X = SymmetricMatrix(N = self.N, d = self.d, dist = dist).X
        
    def compute_labels(self):
        if self.operation == "lanczos":
            self.get_lanczos_tridiag()
        else:
            self.Y = self.operation(self.X)
    
    def from_condition_number(self, cond):
        self.X = SingularvalueMatrix(self.N, self.d, cond).X
        self.cond = cond

    def from_eigenvalues(self, eigenvalues=None, mu=1, sigma=0.2, diagonal=False, similar=False):
        self.X = EigenMatrix(self.N, self.d, eigenvalues, mu, sigma, diagonal, similar).X

    def get_error(self, model):
        id = torch.eye(self.d)
        #This could be generalized to other test errors
        if self.X_with_det is not None: 
            return (torch.matmul(model(self.X_with_det), self.X) - id).square().mean((2, 3)).detach().numpy()
        elif self.X_with_permutations is not None: 
            return (torch.matmul(model(self.X_with_permutations), self.X) - id).square().mean((2, 3)).detach().numpy()
        else:
            return (torch.matmul(model(self.X), self.X) - id).square().mean((2, 3)).detach().numpy()
    
    def compute_determinant(self):
        self.det = torch.linalg.det(self.X)
    
    def compute_cond(self):
        self.cond = torch.linalg.cond(self.X).detach().numpy()
    
    def permute(self,p, dim = 3):
        x = self.X
        x = x[:,:,:,:]
        x_perm = x
        for i in range(p):
            idx = torch.randperm(x.shape[dim]) #dim = 2 for rows, 3 for columns
            x_perm = torch.cat((x_perm,x[:,:,idx,:]),1)
        
        self.X_with_permutations = x_perm
    
    def get_lanczos_tridiag(self, max_iter = 10, device = 'cpu', dtype = torch.float32):
        matrix_shape = self.X[0,0,:,:].shape
        batch_shape = torch.Size([self.N,1])
        init_vecs = torch.ones(matrix_shape[-1], 1, dtype=dtype, device=device)
        init_vecs = init_vecs.expand(*batch_shape, matrix_shape[-1], 1)
        def matmul_closure(v):
            return torch.matmul(self.X,v) 
     
        _,Hm = lanczos.lanczos_tridiag(matmul_closure, max_iter, dtype, device, matrix_shape, batch_shape=torch.Size([self.N,1]))
        self.Y = torch.cat((torch.diagonal(Hm,0,2,3),torch.diagonal(Hm,1,2,3)),2) #torch.sort(torch.linalg.eigh(Hm)[0],2)[0]  
        


class SingularvalueMatrix():
    def __init__(self, N, d=3, cond=10):
        self.N = N
        self.d = d
        self.cond = cond
        self.X = None
        self.matrix_from_singular_values(cond=self.cond)

    def matrix_from_singular_values(self, cond):
        log_cond = np.log(self.cond)
        log_s = torch.arange(-log_cond / 4., log_cond * (self.d + 1) / (4 * (self.d - 1)),
                             log_cond / (2. * (self.d - 1)))
        log_s = log_s.repeat(self.N, 1)
        log_s = log_s[:, :, None]
        s = torch.exp(log_s)
        S = torch.eye(s.shape[1]) * s[:, None]
        U, _ = torch.linalg.qr((torch.rand(self.N, 1, self.d, self.d) - 5.) * 200)
        V, _ = torch.linalg.qr((torch.rand(self.N, 1, self.d, self.d) - 5.) * 200)
        X = torch.matmul(U, torch.matmul(S, V.transpose(2, 3)))
        X = torch.matmul(X, X.transpose(2, 3))
        self.X = X

class SymmetricMatrix():
    def __init__(self,N,d = 5, sigma= 10/math.sqrt(3), dist = "gaussian"):
        self.N = N
        self.d = d
        self.sigma = math.sqrt(self.d)*sigma #Std of Wigner matrices 
        self.dist = dist
        self.X = None
        self.from_distribution(dist = self.dist)
    
    def from_distribution(self, dist = "gaussian"): 
        M = torch.randn(self.N,1,self.d,self.d)
   
        #Create symmetric matrices 
        M = torch.triu(M,0) + torch.transpose(torch.triu(M,1),2,3)  
        #Compute eigenvectors
        P = torch.linalg.eigh(M)[1]
        #Sample new eigenvalues 
        if self.dist == "gaussian":
            x = self.sigma*torch.randn(self.N, self.d, 1)
        elif self.dist == "uniform":
            x = -2*math.sqrt(self.d)*10*torch.rand(self.N, self.d, 1) + math.sqrt(self.d)*10
        elif self.dist == "laplace":
            m = torch.distributions.Laplace(torch.tensor([0.0]), torch.tensor([self.sigma/math.sqrt(2)]))
            x = m.rsample(sample_shape=torch.Size([self.N,self.d]))
        diag = torch.eye(x.shape[1]) * x[:, None]
        #Finally, construct the resulting matrix batch with specified eigenvalues
        self.X = torch.matmul(torch.matmul(P, diag), torch.transpose(P,2,3))


class EigenMatrix():
    def __init__(self, N, d=3, eigenvalues=None, mu= 1, sigma= 0.2, diagonal=False, triangular=False, similar=True):
        self.N = N
        self.d = d
        self.mu = mu
        self.sigma = sigma
        self.eigenvalues = eigenvalues
        self.X = None
        
        if self.eigenvalues != None:
            self.matrix_from_eigenvalues(eigenvalues=self.eigenvalues, diagonal=diagonal, similar=similar)
        else:
            self.matrix_from_eigenvalues(mu=self.mu, sigma=self.sigma, diagonal=diagonal, similar=similar)

    def matrix_from_eigenvalues(self, eigenvalues= None, mu=None, sigma=None, similar=False, diagonal=False):
        #torch.manual_seed(0)
        #Generate N invertible matrices of dimension d. 
        #Eigenvalues for each matrix are sampled from a d IID normal distributions with mean mu and std sigma
        #Resulting matrices are generated via similarity transformatins using random matrices. 
        if eigenvalues != None:
            x = eigenvalues.repeat(self.N, 1)
            x = x[:, :, None]
        else:
            # Set default values if not specified
            #Gaussian 
            if mu == None:
                mu = 0
            elif sigma == None:
                sigma = 0.2
            x = mu * torch.ones((self.N, self.d, 1)) + sigma * torch.randn(self.N, self.d, 1)

            # Create diagonal matrices
        diag = torch.eye(x.shape[1]) * x[:, None]
    
        if not diagonal:
            # Transformation matrix for similarity transform
            if not similar:
                # Creates matrices with different basis.
                M = torch.randn(self.N, 1, self.d, self.d)
                X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
            # Do similarity transform with same basis
            else:
                M = torch.randn(self.d, self.d)
                X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
        else:
            X = diag

        self.X = X
  
