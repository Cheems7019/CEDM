import numpy as np

class TrueSampler_inference_size:
    """
    Sampler for three groups of 10 features each (30 total).

    Y1 (X0-X9):   10 correlated features.
    Y2 (X10-X19): depends on Y1 plus correlated noise;  Y2 = f(Y1) + epsilon_1.
    Y3 (X20-X29): depends on both Y1 and Y2 plus noise; Y3 = g(gamma*Y1, eta*Y2) + epsilon_2.

    gamma and eta control the signal strength from Y1 and Y2 into Y3 respectively.
    """
    def __init__(self, sigma=0.2, correlation=0.5, gamma=1.0, eta=1.0):
        """
        Args:
            sigma: Standard deviation for noise terms
            correlation: Correlation parameter for first slate features
            gamma: Signal level control for Y1 influence on Y3
            eta: Signal level control for Y2 influence on Y3
        """
        self.sigma = sigma
        self.correlation = correlation
        self.gamma = gamma
        self.eta = eta

    def sample(self, n=1000):
        num_features = 30
        slate_size = 10

        # Distance-based correlation: cor(i,j) = correlation^|i-j|.
        correlation_matrix = np.eye(slate_size)
        for i in range(slate_size):
            for j in range(slate_size):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance

        mean = np.zeros(slate_size)
        Y1 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)

        # Noise for Y2 and Y3: correlated within each group, independent across groups.
        epsilon_1 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        epsilon_2 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)

        X = np.zeros((n, num_features))
        X[:, :slate_size] = Y1

        # Y2 = f(Y1) + epsilon_1; nonlinear terms scaled by 0.1 to keep ranges manageable.
        X[:, 10] = 0.1 * (X[:, 0]**3 + X[:, 1]**2 + X[:, 0] * X[:, 1]) + epsilon_1[:, 0]
        X[:, 11] = 0.1 * (X[:, 1]**3 + X[:, 2]**2 + X[:, 1] * X[:, 2]) + epsilon_1[:, 1]
        X[:, 12] = (np.sin(X[:, 0]) + 0.1 * X[:, 2]**2 + X[:, 0] * X[:, 2]) + epsilon_1[:, 2]
        X[:, 13] = (X[:, 3] * X[:, 4] + 0.1 * X[:, 5]**2) + epsilon_1[:, 3]
        X[:, 14] = (np.cos(X[:, 4]) + X[:, 5] * X[:, 6]) + epsilon_1[:, 4]
        X[:, 15] = 0.1 * (X[:, 6]**2 + X[:, 7]**3 + X[:, 6] * X[:, 7]) + epsilon_1[:, 5]
        X[:, 16] = (np.tanh(X[:, 7]) + 0.1 * X[:, 8]**2) + epsilon_1[:, 6]
        X[:, 17] = (X[:, 8] * X[:, 9] + X[:, 0] * X[:, 5]) + epsilon_1[:, 7]
        X[:, 18] = 0.1 * (X[:, 9]**2 + np.sin(X[:, 1]) + X[:, 3] * X[:, 8]) + epsilon_1[:, 8]
        X[:, 19] = (X[:, 0] + X[:, 4] + X[:, 7] + X[:, 9]) + epsilon_1[:, 9]

        # Y3 = g(gamma*Y1, eta*Y2) + epsilon_2.
        # Quadratic terms scaled by 0.5 to limit magnitude.
        Y2 = X[:, 10:20]
        gamma_Y1 = self.gamma * Y1
        eta_Y2 = self.eta * Y2
        X[:, 20] = 0.5 * (gamma_Y1[:, 0]**2 + eta_Y2[:, 0] * gamma_Y1[:, 1] + eta_Y2[:, 1]**2) + epsilon_2[:, 0]
        X[:, 21] = 0.5 * (gamma_Y1[:, 1]**2 + eta_Y2[:, 2]**2 + gamma_Y1[:, 2] * eta_Y2[:, 0]) + epsilon_2[:, 1]
        X[:, 22] = (np.sin(gamma_Y1[:, 0]) + 0.5 * eta_Y2[:, 1]**2 + 0.5 * gamma_Y1[:, 2] * eta_Y2[:, 2]) + epsilon_2[:, 2]
        X[:, 23] = 0.5 * (gamma_Y1[:, 3] * gamma_Y1[:, 4] + eta_Y2[:, 3]**2 + gamma_Y1[:, 5] * eta_Y2[:, 4]) + epsilon_2[:, 3]
        X[:, 24] = (np.cos(gamma_Y1[:, 4]) + 0.5 * eta_Y2[:, 5] * eta_Y2[:, 6] + 0.5 * gamma_Y1[:, 5] * eta_Y2[:, 3]) + epsilon_2[:, 4]
        X[:, 25] = 0.5 * (gamma_Y1[:, 6]**2 + eta_Y2[:, 7]**2 + gamma_Y1[:, 7] * eta_Y2[:, 5]) + epsilon_2[:, 5]
        X[:, 26] = (np.tanh(gamma_Y1[:, 7]) + 0.5 * eta_Y2[:, 8]**2 + 0.5 * gamma_Y1[:, 8] * eta_Y2[:, 6]) + epsilon_2[:, 6]
        X[:, 27] = 0.5 * (gamma_Y1[:, 8] * gamma_Y1[:, 9] + eta_Y2[:, 9] * eta_Y2[:, 0] + gamma_Y1[:, 0] * eta_Y2[:, 7]) + epsilon_2[:, 7]
        X[:, 28] = 0.5 * (gamma_Y1[:, 9]**2 + np.sin(eta_Y2[:, 1]) + gamma_Y1[:, 3] * eta_Y2[:, 8] + eta_Y2[:, 2] * gamma_Y1[:, 4]) + epsilon_2[:, 8]
        X[:, 29] = (gamma_Y1[:, 0] + gamma_Y1[:, 4] + 0.5 * eta_Y2[:, 7] + 0.5 * eta_Y2[:, 9] + 0.5 * gamma_Y1[:, 7] * eta_Y2[:, 3]) + epsilon_2[:, 9]

        return X


class TrueSampler_inference_power:
    """
    Sampler for power analysis with three groups of 10 features each (30 total).

    Y1 (X0-X9):   10 correlated features.
    Y2 (X10-X19): depends on Y1 plus correlated noise; Y2 = f(Y1) + epsilon_1.
    Y3 (X20-X29): depends on both Y1 and Y2 plus noise;
                  Y3 = gamma*g(Y1) + eta*h(Y2) + gamma*eta*f(Y1,Y2) + epsilon_2.

    Every Y3 feature has connections to both Y1 (scaled by gamma) and Y2 (scaled by eta).
    Y2 contributions are variance-balanced relative to Y1 before applying eta.
    """
    def __init__(self, sigma=1.0, correlation=0.5, gamma=1.0, eta=1.0, Y3_noise_scale=1.0):
        """
        Args:
            sigma: Standard deviation for noise terms
            correlation: Correlation parameter for first slate features
            gamma: Signal level control for Y1 influence on Y3
            eta: Signal level control for Y2 influence on Y3
            Y3_noise_scale: Scale factor for noise in Y3 (default=1.0, lower values = less noise)
        """
        self.sigma = sigma
        self.correlation = correlation
        self.gamma = gamma
        self.eta = eta
        self.Y3_noise_scale = Y3_noise_scale

    def sample(self, n=1000):
        num_features = 30
        slate_size = 10

        # Distance-based correlation: cor(i,j) = correlation^|i-j|.
        correlation_matrix = np.eye(slate_size)
        for i in range(slate_size):
            for j in range(slate_size):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance

        mean = np.zeros(slate_size)
        Y1 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)

        X = np.zeros((n, num_features))
        X[:, :slate_size] = Y1

        # Noise for Y2 and Y3: correlated within each group, independent across groups.
        epsilon_1 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        epsilon_2 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)

        # Y2 = f(Y1) + epsilon_1; nonlinear terms scaled by 0.1 to keep ranges manageable.
        X[:, 10] = 0.1 * (Y1[:, 0]**3 + Y1[:, 1]**2 + Y1[:, 0] * Y1[:, 1]) + epsilon_1[:, 0]
        X[:, 11] = 0.1 * (Y1[:, 1]**3 + Y1[:, 2]**2 + Y1[:, 1] * Y1[:, 2]) + epsilon_1[:, 1]
        X[:, 12] = (np.sin(Y1[:, 0]) + 0.1 * Y1[:, 2]**2 + Y1[:, 0] * Y1[:, 2]) + epsilon_1[:, 2]
        X[:, 13] = (Y1[:, 3] * Y1[:, 4] + 0.1 * Y1[:, 5]**2) + epsilon_1[:, 3]
        X[:, 14] = (np.cos(Y1[:, 4]) + Y1[:, 5] * Y1[:, 6]) + epsilon_1[:, 4]
        X[:, 15] = 0.1 * (Y1[:, 6]**2 + Y1[:, 7]**3 + Y1[:, 6] * Y1[:, 7]) + epsilon_1[:, 5]
        X[:, 16] = (np.tanh(Y1[:, 7]) + 0.1 * Y1[:, 8]**2) + epsilon_1[:, 6]
        X[:, 17] = (Y1[:, 8] * Y1[:, 9] + Y1[:, 0] * Y1[:, 5]) + epsilon_1[:, 7]
        X[:, 18] = 0.1 * (Y1[:, 9]**2 + np.sin(Y1[:, 1]) + Y1[:, 3] * Y1[:, 8]) + epsilon_1[:, 8]
        X[:, 19] = (Y1[:, 0] + Y1[:, 4] + Y1[:, 7] + Y1[:, 9]) + epsilon_1[:, 9]

        Y2 = X[:, 10:20]

        # Y2 = f(Y1) + noise has variance ~2*sigma^2 while Y1 has variance sigma^2.
        # Scale Y2 by 1/sqrt(2) so gamma and eta contribute equally to Y3 variance.
        Y2_scale = 1.0 / np.sqrt(2.0)

        # Y3 = gamma*g(Y1) + eta*h(Y2_scaled) + gamma*eta*f(Y1, Y2_scaled) + epsilon_2.
        # Each Y3 feature depends on 5-7 Y1 features and 5-7 Y2 features.
        
        # X20: Depends on Y1[0,1,2,3,4] and Y2[0,1,2,3,4]
        X[:, 20] = (self.gamma * (Y1[:, 0]**2 + Y1[:, 1] + np.sin(Y1[:, 2]) + Y1[:, 3] * Y1[:, 4]) + 
                    self.eta * Y2_scale * (Y2[:, 0]**2 + Y2[:, 1] + Y2[:, 2] * Y2[:, 3] + Y2[:, 4]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 0] * Y2[:, 1] + Y1[:, 3] * Y2[:, 3])) + epsilon_2[:, 0] * self.Y3_noise_scale
        
        # X21: Depends on Y1[1,2,3,4,5] and Y2[1,2,3,4,5]
        X[:, 21] = (self.gamma * (Y1[:, 1]**2 + np.cos(Y1[:, 2]) + Y1[:, 3] + Y1[:, 4] * Y1[:, 5]) + 
                    self.eta * Y2_scale * (Y2[:, 1]**2 + Y2[:, 2] + Y2[:, 3] * Y2[:, 4] + Y2[:, 5]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 2] * Y2[:, 2] + Y1[:, 4] * Y2[:, 4])) + epsilon_2[:, 1] * self.Y3_noise_scale
        
        # X22: Depends on Y1[0,2,3,4,6] and Y2[0,2,3,5,6]
        X[:, 22] = (self.gamma * (np.sin(Y1[:, 0]) + Y1[:, 2]**2 + Y1[:, 3] + np.tanh(Y1[:, 4]) + Y1[:, 6]) + 
                    self.eta * Y2_scale * (Y2[:, 0] + Y2[:, 2]**2 + Y2[:, 3] + Y2[:, 5] * Y2[:, 6]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 3] * Y2[:, 3] + Y1[:, 6] * Y2[:, 5])) + epsilon_2[:, 2] * self.Y3_noise_scale
        
        # X23: Depends on Y1[1,3,4,5,6] and Y2[1,3,4,6,7]
        X[:, 23] = (self.gamma * (Y1[:, 1] + Y1[:, 3] * Y1[:, 4] + Y1[:, 5]**2 + np.cos(Y1[:, 6])) + 
                    self.eta * Y2_scale * (Y2[:, 1] + Y2[:, 3] * Y2[:, 4] + Y2[:, 6]**2 + Y2[:, 7]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 4] * Y2[:, 4] + Y1[:, 6] * Y2[:, 6])) + epsilon_2[:, 3] * self.Y3_noise_scale
        
        # X24: Depends on Y1[0,2,5,7,8] and Y2[2,4,5,7,8]
        X[:, 24] = (self.gamma * (Y1[:, 0]**2 + Y1[:, 2] + np.cos(Y1[:, 5]) + Y1[:, 7] * Y1[:, 8]) + 
                    self.eta * Y2_scale * (Y2[:, 2]**2 + Y2[:, 4] + Y2[:, 5] * Y2[:, 7] + Y2[:, 8]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 5] * Y2[:, 5] + Y1[:, 8] * Y2[:, 7])) + epsilon_2[:, 4] * self.Y3_noise_scale
        
        # X25: Depends on Y1[2,4,6,8,9] and Y2[3,5,6,8,9]
        X[:, 25] = (self.gamma * (Y1[:, 2]**2 + Y1[:, 4] + Y1[:, 6] * Y1[:, 8] + np.tanh(Y1[:, 9])) + 
                    self.eta * Y2_scale * (Y2[:, 3] + Y2[:, 5]**2 + Y2[:, 6] * Y2[:, 8] + Y2[:, 9]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 6] * Y2[:, 6] + Y1[:, 9] * Y2[:, 8])) + epsilon_2[:, 5] * self.Y3_noise_scale
        
        # X26: Depends on Y1[0,3,5,7,9] and Y2[0,4,5,8,9]
        X[:, 26] = (self.gamma * (np.sin(Y1[:, 0]) + Y1[:, 3] + Y1[:, 5] * Y1[:, 7] + np.tanh(Y1[:, 9])) + 
                    self.eta * Y2_scale * (Y2[:, 0]**2 + Y2[:, 4] + Y2[:, 5] * Y2[:, 8] + Y2[:, 9]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 7] * Y2[:, 5] + Y1[:, 9] * Y2[:, 9])) + epsilon_2[:, 6] * self.Y3_noise_scale
        
        # X27: Depends on Y1[1,4,5,7,8,9] and Y2[1,5,6,7,9]
        X[:, 27] = (self.gamma * (Y1[:, 1]**2 + Y1[:, 4] + Y1[:, 5] * Y1[:, 7] + Y1[:, 8] * Y1[:, 9]) + 
                    self.eta * Y2_scale * (Y2[:, 1] + Y2[:, 5]**2 + Y2[:, 6] * Y2[:, 7] + Y2[:, 9]) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 8] * Y2[:, 7] + Y1[:, 9] * Y2[:, 9])) + epsilon_2[:, 7] * self.Y3_noise_scale
        
        # X28: Depends on Y1[0,1,3,6,7,8,9] and Y2[0,2,4,6,8]
        X[:, 28] = (self.gamma * (Y1[:, 0] + Y1[:, 1]**2 + Y1[:, 3] * Y1[:, 6] + Y1[:, 7] + Y1[:, 8] * Y1[:, 9]) + 
                    self.eta * Y2_scale * (Y2[:, 0]**2 + Y2[:, 2] + Y2[:, 4] * Y2[:, 6] + np.sin(Y2[:, 8])) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 3] * Y2[:, 4] + Y1[:, 7] * Y2[:, 6] + Y1[:, 8] * Y2[:, 8])) + epsilon_2[:, 8] * self.Y3_noise_scale
        
        # X29: Depends on Y1[0,2,4,5,6,8,9] and Y2[1,3,5,7,9]
        X[:, 29] = (self.gamma * (Y1[:, 0] + Y1[:, 2] * Y1[:, 4] + Y1[:, 5]**2 + Y1[:, 6] + Y1[:, 8] * Y1[:, 9]) + 
                    self.eta * Y2_scale * (Y2[:, 1] + Y2[:, 3] * Y2[:, 5] + Y2[:, 7] + np.sin(Y2[:, 9])) + 
                    self.gamma * self.eta * Y2_scale * (Y1[:, 0] * Y2[:, 5] + Y1[:, 4] * Y2[:, 3] + Y1[:, 8] * Y2[:, 7])) + epsilon_2[:, 9] * self.Y3_noise_scale
        
        return X


class TrueSampler_sachs:
    """
    Sampler for protein signaling DAG structure with 11 nodes.
    
    Node indices and names:
    0: PKC (root)
    1: Plcg (root)  
    2: PKA (depends on PKC)
    3: PIP3 (depends on Plcg)
    4: Raf (depends on PKC, PKA)
    5: Jnk (depends on PKC, PKA)
    6: P38 (depends on PKC, PKA)
    7: PIP2 (depends on Plcg, PIP3)
    8: Mek (depends on PKC, Raf, PKA)
    9: Erk (depends on Mek, PKA)
    10: Akt (depends on Erk, PKA)
    
    DAG structure (11 nodes, 17 edges):
    - PKC → PKA, Raf, Jnk, P38, Mek (5 edges)
    - Plcg → PIP3, PIP2 (2 edges)
    - PKA → Raf, Jnk, P38, Mek, Erk, Akt (6 edges)
    - PIP3 → PIP2 (1 edge)
    - Raf → Mek (1 edge)
    - Mek → Erk (1 edge)
    - Erk → Akt (1 edge)
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.node_names = ['PKC', 'Plcg', 'PKA', 'PIP3', 'Raf', 'Jnk', 'P38', 'PIP2', 'Mek', 'Erk', 'Akt']
        
    def sample(self, n=1000):
        """Generate n samples following the Sachs DAG (17 edges, 11 nodes).

        Uses compound nonlinearities (tanh of sums of squares, sin of products, etc.)
        with coefficients scaled to prevent variance explosion.
        """
        num_features = 11
        X = np.zeros((n, num_features))
        
        # Root nodes: PKC (0) and Plcg (1).
        X[:, 0] = np.random.normal(0, self.sigma, n)  # PKC
        X[:, 1] = np.random.normal(0, self.sigma, n)  # Plcg

        # Layer 2: PKA = f(PKC), PIP3 = f(Plcg).
        X[:, 2] = (0.5 * np.tanh(X[:, 0]**2) + 0.4 * X[:, 0] +
                   0.3 * np.sin(X[:, 0])**2 + 0.2 * X[:, 0]**2 +
                   np.random.normal(0, self.sigma, n))  # PKA

        X[:, 3] = (0.5 * X[:, 1]**2 + 0.4 * np.sin(X[:, 1]) * np.cos(X[:, 1]) +
                   0.3 * np.tanh(X[:, 1]**2) + 0.2 * X[:, 1] +
                   np.random.normal(0, self.sigma, n))  # PIP3

        # Layer 3: Raf, Jnk, P38 = f(PKC, PKA); PIP2 = f(Plcg, PIP3).
        X[:, 4] = (0.5 * np.tanh(X[:, 0]**2 + X[:, 2]**2) + 0.4 * X[:, 0] * np.sin(X[:, 2]) +
                   0.3 * np.cos(X[:, 0]) * np.tanh(X[:, 2]) + 0.2 * X[:, 0] * X[:, 2] +
                   np.random.normal(0, self.sigma, n))  # Raf

        X[:, 5] = (0.5 * np.tanh(X[:, 0]**2 + X[:, 2]**2) +
                   0.4 * np.sin(X[:, 0] * X[:, 2]) * np.cos(X[:, 2]) +
                   0.3 * np.tanh(X[:, 0]**2) + 0.2 * X[:, 0] * X[:, 2] +
                   np.random.normal(0, self.sigma, n))  # Jnk

        X[:, 6] = (0.5 * np.tanh(X[:, 0]**2) + 0.4 * np.sin(X[:, 2]**2) * np.cos(X[:, 0]) +
                   0.3 * np.tanh(X[:, 0] * X[:, 2]) * np.tanh(X[:, 2]) +
                   0.2 * np.sin(X[:, 0])**2 +
                   np.random.normal(0, self.sigma, n))  # P38

        X[:, 7] = (0.5 * np.tanh(X[:, 1]**2 + X[:, 3]**2) + 0.4 * np.sin(X[:, 1]) * np.tanh(X[:, 3]) +
                   0.3 * np.cos(X[:, 3]) * np.tanh(X[:, 1]) + 0.2 * X[:, 1] * X[:, 3] +
                   np.random.normal(0, self.sigma, n))  # PIP2

        # Layer 4: Mek = f(PKC, Raf, PKA); coefficients slightly reduced for three parents.
        X[:, 8] = (0.4 * np.tanh(X[:, 0]**2 + X[:, 4]**2) +
                   0.4 * np.tanh(X[:, 2] * X[:, 4]) +
                   0.3 * np.sin(X[:, 0]) * np.tanh(X[:, 4] * X[:, 2]) +
                   0.2 * X[:, 0] * X[:, 2] +
                   np.random.normal(0, self.sigma, n))  # Mek

        # Layer 5: Erk = f(Mek, PKA).
        X[:, 9] = (0.5 * np.tanh(X[:, 8]**2 + X[:, 2]**2) +
                   0.4 * np.sin(X[:, 2])**2 * np.tanh(X[:, 8]) +
                   0.3 * np.tanh(X[:, 8] * X[:, 2]) * np.cos(X[:, 2]) +
                   0.2 * np.tanh(X[:, 8]) * X[:, 2] +
                   np.random.normal(0, self.sigma, n))  # Erk

        # Layer 6: Akt = f(Erk, PKA); all bounded terms to prevent variance explosion.
        X[:, 10] = (0.5 * np.tanh(X[:, 9]**2 + X[:, 2]**2) +
                    0.4 * np.cos(X[:, 2])**2 * np.tanh(X[:, 9]) +
                    0.3 * np.tanh(X[:, 9] * X[:, 2]) * np.sin(X[:, 2])**2 +
                    0.2 * np.tanh(X[:, 9]) * np.cos(X[:, 2]) +
                    np.random.normal(0, self.sigma, n))  # Akt
        
        return X
    
    def get_node_index(self, node_name):
        """Get the index of a node by its name."""
        return self.node_names.index(node_name)
    
    def get_node_name(self, index):
        """Get the name of a node by its index."""
        return self.node_names[index]


def compute_conditional_expectation_sachs(plcg_val, pkc_val, pka_val, raf_val, sigma=1.0, n_monte_carlo=100000):
    """
    Compute E[PIP3, Jnk, P38, PIP2, Mek, Erk, Akt | Plcg, PKC, PKA, Raf] via Monte Carlo.
    
    Conditioning variables (given):
    - Plcg (node 1): root node
    - PKC (node 0): root node
    - PKA (node 2): depends on PKC
    - Raf (node 4): depends on PKC, PKA
    
    Target variables (to estimate):
    - PIP3 (node 3): depends on Plcg
    - Jnk (node 5): depends on PKC, PKA
    - P38 (node 6): depends on PKC, PKA
    - PIP2 (node 7): depends on Plcg, PIP3
    - Mek (node 8): depends on PKC, Raf, PKA
    - Erk (node 9): depends on Mek, PKA
    - Akt (node 10): depends on Erk, PKA
    
    Args:
        plcg_val: Value to condition on for Plcg (node 1)
        pkc_val: Value to condition on for PKC (node 0)
        pka_val: Value to condition on for PKA (node 2)
        raf_val: Value to condition on for Raf (node 4)
        sigma: Noise standard deviation (default: 1.0)
        n_monte_carlo: Number of Monte Carlo samples (default: 100000)
    
    Returns:
        Dictionary with conditional means for all 7 target variables
    """
    
    # PIP3 depends only on Plcg (which is conditioned)
    # PIP3 = 0.5*Plcg^2 + 0.4*sin(Plcg)*cos(Plcg) + 0.3*tanh(Plcg^2) + 0.2*Plcg + noise
    pip3_mean = (0.5 * plcg_val**2 + 0.4 * np.sin(plcg_val) * np.cos(plcg_val) + 
                 0.3 * np.tanh(plcg_val**2) + 0.2 * plcg_val)
    
    # Jnk depends on PKC, PKA (both conditioned)
    # Jnk = 0.5*tanh(PKC^2 + PKA^2) + 0.4*sin(PKC*PKA)*cos(PKA) + 0.3*tanh(PKC^2) + 0.2*PKC*PKA + noise
    jnk_mean = (0.5 * np.tanh(pkc_val**2 + pka_val**2) + 
                0.4 * np.sin(pkc_val * pka_val) * np.cos(pka_val) + 
                0.3 * np.tanh(pkc_val**2) + 0.2 * pkc_val * pka_val)
    
    # P38 depends on PKC, PKA (both conditioned)
    # P38 = 0.5*tanh(PKC^2) + 0.4*sin(PKA^2)*cos(PKC) + 0.3*tanh(PKC*PKA)*tanh(PKA) + 0.2*sin(PKC)^2 + noise
    p38_mean = (0.5 * np.tanh(pkc_val**2) + 0.4 * np.sin(pka_val**2) * np.cos(pkc_val) + 
                0.3 * np.tanh(pkc_val * pka_val) * np.tanh(pka_val) + 0.2 * np.sin(pkc_val)**2)
    
    # Mek depends on PKC, Raf, PKA (all conditioned)
    # Mek = 0.4*tanh(PKC^2 + Raf^2) + 0.4*tanh(PKA*Raf) + 0.3*sin(PKC)*tanh(Raf*PKA) + 0.2*PKC*PKA + noise
    mek_mean = (0.4 * np.tanh(pkc_val**2 + raf_val**2) + 
                0.4 * np.tanh(pka_val * raf_val) + 
                0.3 * np.sin(pkc_val) * np.tanh(raf_val * pka_val) + 
                0.2 * pkc_val * pka_val)
    
    # PIP2, Erk, Akt require Monte Carlo: they depend on intermediate nodes with noise.
    pip3_samples = pip3_mean + np.random.normal(0, sigma, n_monte_carlo)

    # PIP2 = f(Plcg, PIP3); Plcg is fixed, PIP3 is sampled above.
    # PIP2 = 0.5*tanh(Plcg^2 + PIP3^2) + 0.4*sin(Plcg)*tanh(PIP3) + 0.3*cos(PIP3)*tanh(Plcg) + 0.2*Plcg*PIP3 + noise
    pip2_samples = (0.5 * np.tanh(plcg_val**2 + pip3_samples**2) + 
                    0.4 * np.sin(plcg_val) * np.tanh(pip3_samples) + 
                    0.3 * np.cos(pip3_samples) * np.tanh(plcg_val) + 
                    0.2 * plcg_val * pip3_samples)
    pip2_mean = np.mean(pip2_samples)

    mek_samples = mek_mean + np.random.normal(0, sigma, n_monte_carlo)

    # Erk = f(Mek, PKA); Mek is sampled, PKA is fixed.
    erk_samples = (0.5 * np.tanh(mek_samples**2 + pka_val**2) +
                   0.4 * np.sin(pka_val)**2 * np.tanh(mek_samples) +
                   0.3 * np.tanh(mek_samples * pka_val) * np.cos(pka_val) +
                   0.2 * np.tanh(mek_samples) * pka_val +
                   np.random.normal(0, sigma, n_monte_carlo))
    erk_mean = np.mean(erk_samples)

    # Akt = f(Erk, PKA); Erk is sampled, PKA is fixed.
    akt_samples = (0.5 * np.tanh(erk_samples**2 + pka_val**2) + 
                   0.4 * np.cos(pka_val)**2 * np.tanh(erk_samples) + 
                   0.3 * np.tanh(erk_samples * pka_val) * np.sin(pka_val)**2 + 
                   0.2 * np.tanh(erk_samples) * np.cos(pka_val))
    akt_mean = np.mean(akt_samples)
    
    return {
        'PIP3': pip3_mean,
        'Jnk': jnk_mean,
        'P38': p38_mean,
        'PIP2': pip2_mean,
        'Mek': mek_mean,
        'Erk': erk_mean,
        'Akt': akt_mean,
        'indices': [3, 5, 6, 7, 8, 9, 10],  # PIP3, Jnk, P38, PIP2, Mek, Erk, Akt
        'names': ['PIP3', 'Jnk', 'P38', 'PIP2', 'Mek', 'Erk', 'Akt']
    }



class TrueSampler_long_chain:
    """
    Sampler for a long chain with 6 slates of 5 nodes each (30 total features).
    Y1 (X0-X4): 5 correlated features
    Y2 (X5-X9): depends on Y1 + correlated noise
    Y3 (X10-X14): depends on Y2 + correlated noise
    Y4 (X15-X19): depends on Y3 + correlated noise
    Y5 (X20-X24): depends on Y4 + correlated noise
    Y6 (X25-X29): depends on Y5 + correlated noise
    
    Chain structure: Y1 → Y2 → Y3 → Y4 → Y5 → Y6
    
    Each slate has correlated noise within itself, but noise is independent across slates.
    Nonlinear functions are controlled to keep all slates at similar scales.
    """
    def __init__(self, sigma=1.0, correlation=0.5):
        """
        Args:
            sigma: Standard deviation for noise terms
            correlation: Correlation parameter for features within each slate
        """
        self.sigma = sigma
        self.correlation = correlation
        
    def sample(self, n=1000):
        """Generate n samples from the long chain structure."""
        num_features = 30  # 6 slates x 5 nodes each
        slate_size = 5
        num_slates = 6

        # Distance-based correlation: cor(i,j) = correlation^|i-j|.
        correlation_matrix = np.eye(slate_size)
        for i in range(slate_size):
            for j in range(slate_size):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance

        X = np.zeros((n, num_features))

        mean = np.zeros(slate_size)
        Y1 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 0:5] = Y1

        # Y2 = f(Y1) + epsilon_2.
        epsilon_2 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 5] = (0.6 * np.tanh(Y1[:, 0]**2 + Y1[:, 1]**2) + 0.5 * Y1[:, 2] * np.sin(Y1[:, 3]) + 
                   0.4 * np.cos(Y1[:, 0]) * np.tanh(Y1[:, 4])) + epsilon_2[:, 0]
        X[:, 6] = (0.6 * Y1[:, 1]**2 + 0.5 * np.sin(Y1[:, 0] * Y1[:, 2]) * np.cos(Y1[:, 3]) + 
                   0.4 * np.tanh(Y1[:, 4]**2)) + epsilon_2[:, 1]
        X[:, 7] = (0.6 * np.tanh(Y1[:, 2]**2) + 0.5 * Y1[:, 3] * Y1[:, 0] + 
                   0.4 * np.sin(Y1[:, 4])**2 * np.cos(Y1[:, 1])) + epsilon_2[:, 2]
        X[:, 8] = (0.6 * Y1[:, 0] * Y1[:, 1] + 0.5 * np.tanh(Y1[:, 2]**2 + Y1[:, 3]**2) + 
                   0.4 * np.sin(Y1[:, 4]) * Y1[:, 2]) + epsilon_2[:, 3]
        X[:, 9] = (0.6 * Y1[:, 4]**2 + 0.5 * np.sin(Y1[:, 1]**2) + 
                   0.4 * np.tanh(Y1[:, 0] * Y1[:, 3]) * np.cos(Y1[:, 2])) + epsilon_2[:, 4]
        
        Y2 = X[:, 5:10]

        # Y3 = f(Y2) + epsilon_3.
        epsilon_3 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 10] = (0.6 * np.tanh(Y2[:, 0]**2 + Y2[:, 1]**2) + 0.5 * np.sin(Y2[:, 2]) * np.tanh(Y2[:, 3]) + 
                    0.4 * np.cos(Y2[:, 4]) * np.tanh(Y2[:, 0])) + epsilon_3[:, 0]
        X[:, 11] = (0.6 * np.tanh(Y2[:, 1]**2) + 0.5 * np.sin(Y2[:, 3]**2) * np.cos(Y2[:, 0]) + 
                    0.4 * np.tanh(Y2[:, 0] * Y2[:, 4]) * np.tanh(Y2[:, 2])) + epsilon_3[:, 1]
        X[:, 12] = (0.6 * np.tanh(Y2[:, 2]**2 + Y2[:, 3]**2) + 0.5 * np.sin(Y2[:, 1]) * np.tanh(Y2[:, 4]) + 
                    0.4 * np.cos(Y2[:, 0])**2) + epsilon_3[:, 2]
        X[:, 13] = (0.6 * np.tanh(Y2[:, 0] * Y2[:, 2]) + 0.5 * np.tanh(Y2[:, 1]**2 + Y2[:, 4]**2) + 
                    0.4 * np.sin(Y2[:, 3]) * np.cos(Y2[:, 2])) + epsilon_3[:, 3]
        X[:, 14] = (0.6 * np.tanh(Y2[:, 4]**2) + 0.5 * np.cos(Y2[:, 1]**2) * np.sin(Y2[:, 3]) + 
                    0.4 * np.tanh(Y2[:, 2] * Y2[:, 3]) * np.tanh(Y2[:, 0])) + epsilon_3[:, 4]
        
        Y3 = X[:, 10:15]

        # Y4 = f(Y3) + epsilon_4.
        epsilon_4 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 15] = (0.6 * np.tanh(Y3[:, 0]**2 + Y3[:, 1]**2) + 0.5 * np.tanh(Y3[:, 2] * Y3[:, 3]) + 
                    0.4 * np.sin(Y3[:, 4]) * np.tanh(Y3[:, 0] * Y3[:, 2])) + epsilon_4[:, 0]
        X[:, 16] = (0.6 * np.tanh(Y3[:, 1]**2) + 0.5 * np.cos(Y3[:, 3])**2 * np.tanh(Y3[:, 4]) + 
                    0.4 * np.tanh(Y3[:, 0] * Y3[:, 4]) * np.sin(Y3[:, 2])) + epsilon_4[:, 1]
        X[:, 17] = (0.6 * np.tanh(Y3[:, 2]**2 + Y3[:, 3]**2) + 0.5 * np.tanh(Y3[:, 1] * Y3[:, 4]) + 
                    0.4 * np.sin(Y3[:, 4])**2 * np.tanh(Y3[:, 0])) + epsilon_4[:, 2]
        X[:, 18] = (0.6 * np.tanh(Y3[:, 0] * Y3[:, 2]) + 0.5 * np.tanh(Y3[:, 1]**2 + Y3[:, 4]**2) + 
                    0.4 * np.cos(Y3[:, 1]) * np.sin(Y3[:, 3]) * np.tanh(Y3[:, 2])) + epsilon_4[:, 3]
        X[:, 19] = (0.6 * np.tanh(Y3[:, 4]**2) + 0.5 * np.sin(Y3[:, 1])**2 * np.tanh(Y3[:, 3]) + 
                    0.4 * np.tanh(Y3[:, 2] * Y3[:, 3]) * np.cos(Y3[:, 0])) + epsilon_4[:, 4]
        
        Y4 = X[:, 15:20]

        # Y5 = f(Y4) + epsilon_5.
        epsilon_5 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 20] = (0.6 * np.tanh(Y4[:, 0]**2 + Y4[:, 1]**2 + Y4[:, 2]**2) + 
                    0.5 * np.tanh(Y4[:, 3]) * np.sin(Y4[:, 4]) + 
                    0.4 * np.cos(Y4[:, 0]) * np.tanh(Y4[:, 1] * Y4[:, 2])) + epsilon_5[:, 0]
        X[:, 21] = (0.6 * np.tanh(Y4[:, 1]**2 + Y4[:, 3]**2) + 
                    0.5 * np.sin(Y4[:, 3])**2 * np.tanh(Y4[:, 0]) + 
                    0.4 * np.tanh(Y4[:, 0] * Y4[:, 4]) * np.cos(Y4[:, 2])) + epsilon_5[:, 1]
        X[:, 22] = (0.6 * np.tanh(Y4[:, 2]**2 + Y4[:, 4]**2) + 
                    0.5 * np.tanh(Y4[:, 3]) * np.cos(Y4[:, 1]) + 
                    0.4 * np.cos(Y4[:, 4])**2 * np.tanh(Y4[:, 0] * Y4[:, 2])) + epsilon_5[:, 2]
        X[:, 23] = (0.6 * np.tanh(Y4[:, 0] * Y4[:, 2] + Y4[:, 1] * Y4[:, 4]) + 
                    0.5 * np.tanh(Y4[:, 2]**2) * np.sin(Y4[:, 3]) + 
                    0.4 * np.sin(Y4[:, 1]) * np.tanh(Y4[:, 4])) + epsilon_5[:, 3]
        X[:, 24] = (0.6 * np.tanh(Y4[:, 4]**2 + Y4[:, 0]**2) + 
                    0.5 * np.cos(Y4[:, 1])**2 * np.tanh(Y4[:, 3]) + 
                    0.4 * np.tanh(Y4[:, 2] * Y4[:, 3]) * np.sin(Y4[:, 0])) + epsilon_5[:, 4]
        
        Y5 = X[:, 20:25]

        # Y6 = f(Y5) + epsilon_6.
        epsilon_6 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 25] = (0.6 * np.tanh(Y5[:, 0]**2 + Y5[:, 1]**2 + Y5[:, 2]**2) + 
                    0.5 * np.tanh(Y5[:, 1]) * np.sin(Y5[:, 2])**2 + 
                    0.4 * np.tanh(Y5[:, 3] * Y5[:, 4]) * np.cos(Y5[:, 0])) + epsilon_6[:, 0]
        X[:, 26] = (0.6 * np.tanh(Y5[:, 1]**2 + Y5[:, 3]**2) + 
                    0.5 * np.cos(Y5[:, 3])**2 * np.tanh(Y5[:, 0] + Y5[:, 4]) + 
                    0.4 * np.tanh(Y5[:, 0] * Y5[:, 4]) * np.sin(Y5[:, 2])**2) + epsilon_6[:, 1]
        X[:, 27] = (0.6 * np.tanh(Y5[:, 2]**2 + Y5[:, 3]**2 + Y5[:, 4]**2) + 
                    0.5 * np.tanh(Y5[:, 3]) * np.cos(Y5[:, 1]) + 
                    0.4 * np.sin(Y5[:, 4])**2 * np.tanh(Y5[:, 0] * Y5[:, 2])) + epsilon_6[:, 2]
        X[:, 28] = (0.6 * np.tanh(Y5[:, 0] * Y5[:, 2] + Y5[:, 1] * Y5[:, 4]) + 
                    0.5 * np.tanh(Y5[:, 2]**2) * np.sin(Y5[:, 3]) * np.cos(Y5[:, 4]) + 
                    0.4 * np.cos(Y5[:, 1])**2 * np.tanh(Y5[:, 0])) + epsilon_6[:, 3]
        X[:, 29] = (0.6 * np.tanh(Y5[:, 4]**2 + Y5[:, 0]**2 + Y5[:, 1]**2) + 
                    0.5 * np.sin(Y5[:, 1])**2 * np.tanh(Y5[:, 3] * Y5[:, 4]) + 
                    0.4 * np.tanh(Y5[:, 2] * Y5[:, 3]) * np.cos(Y5[:, 0])**2) + epsilon_6[:, 4]
        
        return X


class TrueSampler_hub:
    """
    Sampler for a hub structure with 6 slates of 5 nodes each (30 total features).
    Y1 (X0-X4): 5 correlated features (hub/root)
    Y2 (X5-X9): depends on Y1 + correlated noise
    Y3 (X10-X14): depends on Y1 + correlated noise
    Y4 (X15-X19): depends on Y1 + correlated noise
    Y5 (X20-X24): depends on Y1 + correlated noise
    Y6 (X25-X29): depends on Y1 + correlated noise
    
    Hub structure: Y1 → Y2, Y1 → Y3, Y1 → Y4, Y1 → Y5, Y1 → Y6
    All slates Y2-Y6 depend directly on Y1 (no dependencies between Y2-Y6).
    
    Each slate has correlated noise within itself, but noise is independent across slates.
    Nonlinear functions are controlled to keep all slates at similar scales.
    """
    def __init__(self, sigma=1.0, correlation=0.5):
        """
        Args:
            sigma: Standard deviation for noise terms
            correlation: Correlation parameter for features within each slate
        """
        self.sigma = sigma
        self.correlation = correlation
        
    def sample(self, n=1000):
        """Generate n samples from the hub structure."""
        num_features = 30  # 6 slates x 5 nodes each
        slate_size = 5

        # Distance-based correlation: cor(i,j) = correlation^|i-j|.
        correlation_matrix = np.eye(slate_size)
        for i in range(slate_size):
            for j in range(slate_size):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance

        X = np.zeros((n, num_features))

        mean = np.zeros(slate_size)
        Y1 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 0:5] = Y1

        # Y2 = f(Y1) + epsilon_2.
        epsilon_2 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 5] = (0.8 * np.tanh(Y1[:, 0]**2) + 0.6 * Y1[:, 1]**2 + 0.5 * np.sin(Y1[:, 2]) * np.cos(Y1[:, 3])) + epsilon_2[:, 0]
        X[:, 6] = (0.7 * Y1[:, 1]**2 + 0.6 * np.cos(Y1[:, 3])**2 + 0.5 * Y1[:, 0] * Y1[:, 4]**2) + epsilon_2[:, 1]
        X[:, 7] = (0.8 * np.tanh(Y1[:, 2])**2 + 0.6 * Y1[:, 3]**2 + 0.5 * np.sin(Y1[:, 4]) * Y1[:, 0]) + epsilon_2[:, 2]
        X[:, 8] = (0.7 * Y1[:, 0]**2 + 0.6 * Y1[:, 2] * Y1[:, 4] + 0.5 * np.tanh(Y1[:, 1]**2 + Y1[:, 3]**2)) + epsilon_2[:, 3]
        X[:, 9] = (0.8 * Y1[:, 4]**2 + 0.6 * np.sin(Y1[:, 1])**2 + 0.5 * Y1[:, 2]**2 * np.tanh(Y1[:, 3])) + epsilon_2[:, 4]
        
        # Y3 = f(Y1) + epsilon_3.
        epsilon_3 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 10] = (0.8 * np.sin(Y1[:, 0])**2 + 0.6 * Y1[:, 2]**2 + 0.5 * np.cos(Y1[:, 1]) * Y1[:, 4]) + epsilon_3[:, 0]
        X[:, 11] = (0.7 * Y1[:, 2]**2 + 0.6 * np.tanh(Y1[:, 4])**2 + 0.5 * Y1[:, 1]**2 * Y1[:, 3]) + epsilon_3[:, 1]
        X[:, 12] = (0.8 * np.cos(Y1[:, 3])**2 + 0.6 * Y1[:, 0]**2 + 0.5 * np.tanh(Y1[:, 4]) * Y1[:, 2]) + epsilon_3[:, 2]
        X[:, 13] = (0.7 * Y1[:, 1]**2 + 0.6 * Y1[:, 3] * Y1[:, 4] + 0.5 * np.sin(Y1[:, 0]**2) * np.cos(Y1[:, 2])) + epsilon_3[:, 3]
        X[:, 14] = (0.8 * Y1[:, 3]**2 + 0.6 * np.cos(Y1[:, 2])**2 + 0.5 * Y1[:, 0] * Y1[:, 4]**2) + epsilon_3[:, 4]
        
        # Y4 = f(Y1) + epsilon_4.
        epsilon_4 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 15] = (0.8 * np.tanh(Y1[:, 1]**2) + 0.6 * Y1[:, 3]**2 + 0.5 * np.cos(Y1[:, 0]) * Y1[:, 4]) + epsilon_4[:, 0]
        X[:, 16] = (0.7 * Y1[:, 4]**2 + 0.6 * np.sin(Y1[:, 2])**2 + 0.5 * Y1[:, 1] * Y1[:, 3]**2) + epsilon_4[:, 1]
        X[:, 17] = (0.8 * np.sin(Y1[:, 3])**2 + 0.6 * Y1[:, 1]**2 + 0.5 * np.tanh(Y1[:, 2]) * Y1[:, 0]) + epsilon_4[:, 2]
        X[:, 18] = (0.7 * Y1[:, 2]**2 + 0.6 * Y1[:, 4] * Y1[:, 0] + 0.5 * np.cos(Y1[:, 0]**2) * np.sin(Y1[:, 1])) + epsilon_4[:, 3]
        X[:, 19] = (0.8 * Y1[:, 0]**2 + 0.6 * np.tanh(Y1[:, 3])**2 + 0.5 * Y1[:, 2] * Y1[:, 4]**2) + epsilon_4[:, 4]
        
        # Y5 = f(Y1) + epsilon_5.
        epsilon_5 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 20] = (0.8 * np.cos(Y1[:, 2])**2 + 0.6 * Y1[:, 4]**2 + 0.5 * np.sin(Y1[:, 1]) * Y1[:, 3]) + epsilon_5[:, 0]
        X[:, 21] = (0.7 * Y1[:, 3]**2 + 0.6 * np.tanh(Y1[:, 0])**2 + 0.5 * Y1[:, 2]**2 * Y1[:, 4]) + epsilon_5[:, 1]
        X[:, 22] = (0.8 * np.tanh(Y1[:, 4])**2 + 0.6 * Y1[:, 2]**2 + 0.5 * np.cos(Y1[:, 3]) * Y1[:, 1]) + epsilon_5[:, 2]
        X[:, 23] = (0.7 * Y1[:, 0]**2 + 0.6 * Y1[:, 1] * Y1[:, 4] + 0.5 * np.sin(Y1[:, 3]**2) * np.tanh(Y1[:, 2])) + epsilon_5[:, 3]
        X[:, 24] = (0.8 * Y1[:, 1]**2 + 0.6 * np.sin(Y1[:, 4])**2 + 0.5 * Y1[:, 0] * Y1[:, 3]**2) + epsilon_5[:, 4]
        
        # Y6 = f(Y1) + epsilon_6.
        epsilon_6 = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        X[:, 25] = (0.8 * np.sin(Y1[:, 4])**2 + 0.6 * Y1[:, 0]**2 + 0.5 * np.tanh(Y1[:, 2]) * Y1[:, 1]) + epsilon_6[:, 0]
        X[:, 26] = (0.7 * Y1[:, 2]**2 + 0.6 * np.cos(Y1[:, 1])**2 + 0.5 * Y1[:, 3] * Y1[:, 4]**2) + epsilon_6[:, 1]
        X[:, 27] = (0.8 * np.tanh(Y1[:, 3])**2 + 0.6 * Y1[:, 4]**2 + 0.5 * np.sin(Y1[:, 0]) * Y1[:, 2]) + epsilon_6[:, 2]
        X[:, 28] = (0.7 * Y1[:, 1]**2 + 0.6 * Y1[:, 3] * Y1[:, 0] + 0.5 * np.cos(Y1[:, 2]**2) * np.tanh(Y1[:, 4])) + epsilon_6[:, 3]
        X[:, 29] = (0.8 * Y1[:, 4]**2 + 0.6 * np.tanh(Y1[:, 0])**2 + 0.5 * Y1[:, 1]**2 * Y1[:, 2]) + epsilon_6[:, 4]
        
        return X


def compute_conditional_expectation_hub(Y1_val, sigma=1.0, correlation=0.5):
    """
    Compute the conditional expectation E[Y2-Y6 | Y1 = Y1_val] for hub structure.
    
    Since Y2-Y6 all depend only on Y1 (and not on each other), the conditional
    expectations can be computed directly from the deterministic functions.
    
    Args:
        Y1_val: (5,) array-like, values to condition Y1 on
        sigma: Standard deviation for noise terms (default: 1.0)
        correlation: Correlation parameter for features within each slate (default: 0.5)
    
    Returns:
        Dictionary with conditional means of Y2-Y6 (5×5=25-dimensional) given Y1
    
    Note: Since Yi = f(Y1) + epsilon for i=2,...,6, we have E[Yi | Y1] = f(Y1).
    No Monte Carlo needed - just evaluate the deterministic functions.
    """
    Y1_val = np.asarray(Y1_val)
    if Y1_val.shape[0] != 5:
        raise ValueError("Y1_val must be a 5-dimensional vector")
    
    slate_size = 5
    
    # E[Yi | Y1] = deterministic part of f_i(Y1), since noise mean is zero.
    Y2_mean = np.zeros(slate_size)
    Y2_mean[0] = 0.8 * np.tanh(Y1_val[0]**2) + 0.6 * Y1_val[1]**2 + 0.5 * np.sin(Y1_val[2]) * np.cos(Y1_val[3])
    Y2_mean[1] = 0.7 * Y1_val[1]**2 + 0.6 * np.cos(Y1_val[3])**2 + 0.5 * Y1_val[0] * Y1_val[4]**2
    Y2_mean[2] = 0.8 * np.tanh(Y1_val[2])**2 + 0.6 * Y1_val[3]**2 + 0.5 * np.sin(Y1_val[4]) * Y1_val[0]
    Y2_mean[3] = 0.7 * Y1_val[0]**2 + 0.6 * Y1_val[2] * Y1_val[4] + 0.5 * np.tanh(Y1_val[1]**2 + Y1_val[3]**2)
    Y2_mean[4] = 0.8 * Y1_val[4]**2 + 0.6 * np.sin(Y1_val[1])**2 + 0.5 * Y1_val[2]**2 * np.tanh(Y1_val[3])
    
    # Y3 (X10-X14):
    Y3_mean = np.zeros(slate_size)
    Y3_mean[0] = 0.8 * np.sin(Y1_val[0])**2 + 0.6 * Y1_val[2]**2 + 0.5 * np.cos(Y1_val[1]) * Y1_val[4]
    Y3_mean[1] = 0.7 * Y1_val[2]**2 + 0.6 * np.tanh(Y1_val[4])**2 + 0.5 * Y1_val[1]**2 * Y1_val[3]
    Y3_mean[2] = 0.8 * np.cos(Y1_val[3])**2 + 0.6 * Y1_val[0]**2 + 0.5 * np.tanh(Y1_val[4]) * Y1_val[2]
    Y3_mean[3] = 0.7 * Y1_val[1]**2 + 0.6 * Y1_val[3] * Y1_val[4] + 0.5 * np.sin(Y1_val[0]**2) * np.cos(Y1_val[2])
    Y3_mean[4] = 0.8 * Y1_val[3]**2 + 0.6 * np.cos(Y1_val[2])**2 + 0.5 * Y1_val[0] * Y1_val[4]**2
    
    # Y4 (X15-X19):
    Y4_mean = np.zeros(slate_size)
    Y4_mean[0] = 0.8 * np.tanh(Y1_val[1]**2) + 0.6 * Y1_val[3]**2 + 0.5 * np.cos(Y1_val[0]) * Y1_val[4]
    Y4_mean[1] = 0.7 * Y1_val[4]**2 + 0.6 * np.sin(Y1_val[2])**2 + 0.5 * Y1_val[1] * Y1_val[3]**2
    Y4_mean[2] = 0.8 * np.sin(Y1_val[3])**2 + 0.6 * Y1_val[1]**2 + 0.5 * np.tanh(Y1_val[2]) * Y1_val[0]
    Y4_mean[3] = 0.7 * Y1_val[2]**2 + 0.6 * Y1_val[4] * Y1_val[0] + 0.5 * np.cos(Y1_val[0]**2) * np.sin(Y1_val[1])
    Y4_mean[4] = 0.8 * Y1_val[0]**2 + 0.6 * np.tanh(Y1_val[3])**2 + 0.5 * Y1_val[2] * Y1_val[4]**2
    
    # Y5 (X20-X24):
    Y5_mean = np.zeros(slate_size)
    Y5_mean[0] = 0.8 * np.cos(Y1_val[2])**2 + 0.6 * Y1_val[4]**2 + 0.5 * np.sin(Y1_val[1]) * Y1_val[3]
    Y5_mean[1] = 0.7 * Y1_val[3]**2 + 0.6 * np.tanh(Y1_val[0])**2 + 0.5 * Y1_val[2]**2 * Y1_val[4]
    Y5_mean[2] = 0.8 * np.tanh(Y1_val[4])**2 + 0.6 * Y1_val[2]**2 + 0.5 * np.cos(Y1_val[3]) * Y1_val[1]
    Y5_mean[3] = 0.7 * Y1_val[0]**2 + 0.6 * Y1_val[1] * Y1_val[4] + 0.5 * np.sin(Y1_val[3]**2) * np.tanh(Y1_val[2])
    Y5_mean[4] = 0.8 * Y1_val[1]**2 + 0.6 * np.sin(Y1_val[4])**2 + 0.5 * Y1_val[0] * Y1_val[3]**2
    
    # Y6 (X25-X29):
    Y6_mean = np.zeros(slate_size)
    Y6_mean[0] = 0.8 * np.sin(Y1_val[4])**2 + 0.6 * Y1_val[0]**2 + 0.5 * np.tanh(Y1_val[2]) * Y1_val[1]
    Y6_mean[1] = 0.7 * Y1_val[2]**2 + 0.6 * np.cos(Y1_val[1])**2 + 0.5 * Y1_val[3] * Y1_val[4]**2
    Y6_mean[2] = 0.8 * np.tanh(Y1_val[3])**2 + 0.6 * Y1_val[4]**2 + 0.5 * np.sin(Y1_val[0]) * Y1_val[2]
    Y6_mean[3] = 0.7 * Y1_val[1]**2 + 0.6 * Y1_val[3] * Y1_val[0] + 0.5 * np.cos(Y1_val[2]**2) * np.tanh(Y1_val[4])
    Y6_mean[4] = 0.8 * Y1_val[4]**2 + 0.6 * np.tanh(Y1_val[0])**2 + 0.5 * Y1_val[1]**2 * Y1_val[2]
    
    return {
        'Y2_mean': Y2_mean,
        'Y3_mean': Y3_mean,
        'Y4_mean': Y4_mean,
        'Y5_mean': Y5_mean,
        'Y6_mean': Y6_mean,
        'Y6_mean_only': Y6_mean,  # For compatibility with comparison script
        'indices_Y6': list(range(25, 30)),  # X25-X29
        'names_Y6': [f'X{i}' for i in range(25, 30)]
    }


def compute_conditional_expectation_long_chain(Y1_val, sigma=1.0, correlation=0.5, n_monte_carlo=100000):
    """
    Compute the conditional expectation E[Y6 | Y1 = Y1_val] using Monte Carlo sampling.
    
    Args:
        Y1_val: (5,) array-like, values to condition Y1 on
        sigma: Standard deviation for noise terms (default: 1.0)
        correlation: Correlation parameter for features within each slate (default: 0.5)
        n_monte_carlo: Number of Monte Carlo samples for expectation estimation (default: 100000)
    
    Returns:
        Dictionary with conditional mean of Y6 (5-dimensional vector) given Y1
    
    Note: Since the chain has nonlinear relationships, E[Y6 | Y1] ≠ f(Y1).
    We propagate uncertainty through the chain by sampling:
    Y1 → Y2 → Y3 → Y4 → Y5 → Y6
    """
    Y1_val = np.asarray(Y1_val)
    if Y1_val.shape[0] != 5:
        raise ValueError("Y1_val must be a 5-dimensional vector")
    
    slate_size = 5
    
    correlation_matrix = np.eye(slate_size)
    for i in range(slate_size):
        for j in range(slate_size):
            if i != j:
                distance = abs(i - j)
                correlation_matrix[i, j] = correlation ** distance

    mean = np.zeros(slate_size)

    # Propagate through chain Y1 -> Y2 -> ... -> Y6, sampling noise at each step.
    epsilon_2 = np.random.multivariate_normal(mean, correlation_matrix * sigma**2, size=n_monte_carlo)
    Y2_samples = np.zeros((n_monte_carlo, slate_size))
    Y2_samples[:, 0] = (0.6 * np.tanh(Y1_val[0]**2 + Y1_val[1]**2) + 0.5 * Y1_val[2] * np.sin(Y1_val[3]) + 
                        0.4 * np.cos(Y1_val[0]) * np.tanh(Y1_val[4])) + epsilon_2[:, 0]
    Y2_samples[:, 1] = (0.6 * Y1_val[1]**2 + 0.5 * np.sin(Y1_val[0] * Y1_val[2]) * np.cos(Y1_val[3]) + 
                        0.4 * np.tanh(Y1_val[4]**2)) + epsilon_2[:, 1]
    Y2_samples[:, 2] = (0.6 * np.tanh(Y1_val[2]**2) + 0.5 * Y1_val[3] * Y1_val[0] + 
                        0.4 * np.sin(Y1_val[4])**2 * np.cos(Y1_val[1])) + epsilon_2[:, 2]
    Y2_samples[:, 3] = (0.6 * Y1_val[0] * Y1_val[1] + 0.5 * np.tanh(Y1_val[2]**2 + Y1_val[3]**2) + 
                        0.4 * np.sin(Y1_val[4]) * Y1_val[2]) + epsilon_2[:, 3]
    Y2_samples[:, 4] = (0.6 * Y1_val[4]**2 + 0.5 * np.sin(Y1_val[1]**2) + 
                        0.4 * np.tanh(Y1_val[0] * Y1_val[3]) * np.cos(Y1_val[2])) + epsilon_2[:, 4]
    
    epsilon_3 = np.random.multivariate_normal(mean, correlation_matrix * sigma**2, size=n_monte_carlo)
    Y3_samples = np.zeros((n_monte_carlo, slate_size))
    Y3_samples[:, 0] = (0.6 * np.tanh(Y2_samples[:, 0]**2 + Y2_samples[:, 1]**2) + 0.5 * np.sin(Y2_samples[:, 2]) * np.tanh(Y2_samples[:, 3]) + 
                        0.4 * np.cos(Y2_samples[:, 4]) * np.tanh(Y2_samples[:, 0])) + epsilon_3[:, 0]
    Y3_samples[:, 1] = (0.6 * np.tanh(Y2_samples[:, 1]**2) + 0.5 * np.sin(Y2_samples[:, 3]**2) * np.cos(Y2_samples[:, 0]) + 
                        0.4 * np.tanh(Y2_samples[:, 0] * Y2_samples[:, 4]) * np.tanh(Y2_samples[:, 2])) + epsilon_3[:, 1]
    Y3_samples[:, 2] = (0.6 * np.tanh(Y2_samples[:, 2]**2 + Y2_samples[:, 3]**2) + 0.5 * np.sin(Y2_samples[:, 1]) * np.tanh(Y2_samples[:, 4]) + 
                        0.4 * np.cos(Y2_samples[:, 0])**2) + epsilon_3[:, 2]
    Y3_samples[:, 3] = (0.6 * np.tanh(Y2_samples[:, 0] * Y2_samples[:, 2]) + 0.5 * np.tanh(Y2_samples[:, 1]**2 + Y2_samples[:, 4]**2) + 
                        0.4 * np.sin(Y2_samples[:, 3]) * np.cos(Y2_samples[:, 2])) + epsilon_3[:, 3]
    Y3_samples[:, 4] = (0.6 * np.tanh(Y2_samples[:, 4]**2) + 0.5 * np.cos(Y2_samples[:, 1]**2) * np.sin(Y2_samples[:, 3]) + 
                        0.4 * np.tanh(Y2_samples[:, 2] * Y2_samples[:, 3]) * np.tanh(Y2_samples[:, 0])) + epsilon_3[:, 4]
    
    epsilon_4 = np.random.multivariate_normal(mean, correlation_matrix * sigma**2, size=n_monte_carlo)
    Y4_samples = np.zeros((n_monte_carlo, slate_size))
    Y4_samples[:, 0] = (0.6 * np.tanh(Y3_samples[:, 0]**2 + Y3_samples[:, 1]**2) + 0.5 * np.tanh(Y3_samples[:, 2] * Y3_samples[:, 3]) + 
                        0.4 * np.sin(Y3_samples[:, 4]) * np.tanh(Y3_samples[:, 0] * Y3_samples[:, 2])) + epsilon_4[:, 0]
    Y4_samples[:, 1] = (0.6 * np.tanh(Y3_samples[:, 1]**2) + 0.5 * np.cos(Y3_samples[:, 3])**2 * np.tanh(Y3_samples[:, 4]) + 
                        0.4 * np.tanh(Y3_samples[:, 0] * Y3_samples[:, 4]) * np.sin(Y3_samples[:, 2])) + epsilon_4[:, 1]
    Y4_samples[:, 2] = (0.6 * np.tanh(Y3_samples[:, 2]**2 + Y3_samples[:, 3]**2) + 0.5 * np.tanh(Y3_samples[:, 1] * Y3_samples[:, 4]) + 
                        0.4 * np.sin(Y3_samples[:, 4])**2 * np.tanh(Y3_samples[:, 0])) + epsilon_4[:, 2]
    Y4_samples[:, 3] = (0.6 * np.tanh(Y3_samples[:, 0] * Y3_samples[:, 2]) + 0.5 * np.tanh(Y3_samples[:, 1]**2 + Y3_samples[:, 4]**2) + 
                        0.4 * np.cos(Y3_samples[:, 1]) * np.sin(Y3_samples[:, 3]) * np.tanh(Y3_samples[:, 2])) + epsilon_4[:, 3]
    Y4_samples[:, 4] = (0.6 * np.tanh(Y3_samples[:, 4]**2) + 0.5 * np.sin(Y3_samples[:, 1])**2 * np.tanh(Y3_samples[:, 3]) + 
                        0.4 * np.tanh(Y3_samples[:, 2] * Y3_samples[:, 3]) * np.cos(Y3_samples[:, 0])) + epsilon_4[:, 4]
    
    epsilon_5 = np.random.multivariate_normal(mean, correlation_matrix * sigma**2, size=n_monte_carlo)
    Y5_samples = np.zeros((n_monte_carlo, slate_size))
    Y5_samples[:, 0] = (0.6 * np.tanh(Y4_samples[:, 0]**2 + Y4_samples[:, 1]**2 + Y4_samples[:, 2]**2) + 
                        0.5 * np.tanh(Y4_samples[:, 3]) * np.sin(Y4_samples[:, 4]) + 
                        0.4 * np.cos(Y4_samples[:, 0]) * np.tanh(Y4_samples[:, 1] * Y4_samples[:, 2])) + epsilon_5[:, 0]
    Y5_samples[:, 1] = (0.6 * np.tanh(Y4_samples[:, 1]**2 + Y4_samples[:, 3]**2) + 
                        0.5 * np.sin(Y4_samples[:, 3])**2 * np.tanh(Y4_samples[:, 0]) + 
                        0.4 * np.tanh(Y4_samples[:, 0] * Y4_samples[:, 4]) * np.cos(Y4_samples[:, 2])) + epsilon_5[:, 1]
    Y5_samples[:, 2] = (0.6 * np.tanh(Y4_samples[:, 2]**2 + Y4_samples[:, 4]**2) + 
                        0.5 * np.tanh(Y4_samples[:, 3]) * np.cos(Y4_samples[:, 1]) + 
                        0.4 * np.cos(Y4_samples[:, 4])**2 * np.tanh(Y4_samples[:, 0] * Y4_samples[:, 2])) + epsilon_5[:, 2]
    Y5_samples[:, 3] = (0.6 * np.tanh(Y4_samples[:, 0] * Y4_samples[:, 2] + Y4_samples[:, 1] * Y4_samples[:, 4]) + 
                        0.5 * np.tanh(Y4_samples[:, 2]**2) * np.sin(Y4_samples[:, 3]) + 
                        0.4 * np.sin(Y4_samples[:, 1]) * np.tanh(Y4_samples[:, 4])) + epsilon_5[:, 3]
    Y5_samples[:, 4] = (0.6 * np.tanh(Y4_samples[:, 4]**2 + Y4_samples[:, 0]**2) + 
                        0.5 * np.cos(Y4_samples[:, 1])**2 * np.tanh(Y4_samples[:, 3]) + 
                        0.4 * np.tanh(Y4_samples[:, 2] * Y4_samples[:, 3]) * np.sin(Y4_samples[:, 0])) + epsilon_5[:, 4]
    
    # Y6 deterministic part only (noise mean is zero, so E[Y6|Y1] = E[f(Y5)]).
    Y6_samples = np.zeros((n_monte_carlo, slate_size))
    Y6_samples[:, 0] = (0.6 * np.tanh(Y5_samples[:, 0]**2 + Y5_samples[:, 1]**2 + Y5_samples[:, 2]**2) + 
                        0.5 * np.tanh(Y5_samples[:, 1]) * np.sin(Y5_samples[:, 2])**2 + 
                        0.4 * np.tanh(Y5_samples[:, 3] * Y5_samples[:, 4]) * np.cos(Y5_samples[:, 0]))
    Y6_samples[:, 1] = (0.6 * np.tanh(Y5_samples[:, 1]**2 + Y5_samples[:, 3]**2) + 
                        0.5 * np.cos(Y5_samples[:, 3])**2 * np.tanh(Y5_samples[:, 0] + Y5_samples[:, 4]) + 
                        0.4 * np.tanh(Y5_samples[:, 0] * Y5_samples[:, 4]) * np.sin(Y5_samples[:, 2])**2)
    Y6_samples[:, 2] = (0.6 * np.tanh(Y5_samples[:, 2]**2 + Y5_samples[:, 3]**2 + Y5_samples[:, 4]**2) + 
                        0.5 * np.tanh(Y5_samples[:, 3]) * np.cos(Y5_samples[:, 1]) + 
                        0.4 * np.sin(Y5_samples[:, 4])**2 * np.tanh(Y5_samples[:, 0] * Y5_samples[:, 2]))
    Y6_samples[:, 3] = (0.6 * np.tanh(Y5_samples[:, 0] * Y5_samples[:, 2] + Y5_samples[:, 1] * Y5_samples[:, 4]) + 
                        0.5 * np.tanh(Y5_samples[:, 2]**2) * np.sin(Y5_samples[:, 3]) * np.cos(Y5_samples[:, 4]) + 
                        0.4 * np.cos(Y5_samples[:, 1])**2 * np.tanh(Y5_samples[:, 0]))
    Y6_samples[:, 4] = (0.6 * np.tanh(Y5_samples[:, 4]**2 + Y5_samples[:, 0]**2 + Y5_samples[:, 1]**2) + 
                        0.5 * np.sin(Y5_samples[:, 1])**2 * np.tanh(Y5_samples[:, 3] * Y5_samples[:, 4]) + 
                        0.4 * np.tanh(Y5_samples[:, 2] * Y5_samples[:, 3]) * np.cos(Y5_samples[:, 0])**2)
    
    Y2_mean = np.mean(Y2_samples, axis=0)
    Y3_mean = np.mean(Y3_samples, axis=0)
    Y4_mean = np.mean(Y4_samples, axis=0)
    Y5_mean = np.mean(Y5_samples, axis=0)
    Y6_mean = np.mean(Y6_samples, axis=0)
    
    return {
        'Y2_mean': Y2_mean,
        'Y3_mean': Y3_mean,
        'Y4_mean': Y4_mean,
        'Y5_mean': Y5_mean,
        'Y6_mean': Y6_mean,
        'Y2_std': np.std(Y2_samples, axis=0),
        'Y3_std': np.std(Y3_samples, axis=0),
        'Y4_std': np.std(Y4_samples, axis=0),
        'Y5_std': np.std(Y5_samples, axis=0),
        'Y6_std': np.std(Y6_samples, axis=0),
        'indices': list(range(5, 30)),  # X5-X29 (all Y2-Y6)
        'names': [f'X{i}' for i in range(5, 30)]
    }


def sample_random_dag(n_slates=6, edge_prob=0.5, seed=None):
    """
    Sample a random DAG at the slate level with topological ordering Y1 -> ... -> Y6.
    
    Args:
        n_slates: Number of slates (default: 6)
        edge_prob: Probability of edge Yi -> Yj for any i < j (default: 0.5)
        seed: Random seed for reproducibility
    
    Returns:
        A: (n_slates, n_slates) binary adjacency matrix where A[i,j] = 1 means Yi -> Yj
           Guarantees topological order: edges only from lower index to higher index
    """
    if seed is not None:
        np.random.seed(seed)
    
    A = np.zeros((n_slates, n_slates), dtype=int)
    
    # Sample edges only from i to j where i < j (ensures DAG with topological order)
    for i in range(n_slates):
        for j in range(i + 1, n_slates):
            if np.random.random() < edge_prob:
                A[i, j] = 1
    
    return A


class TrueSampler_random:
    """
    Sampler for 30 variables in 6 slates (5 nodes each) under an arbitrary DAG.

    Each node in slate Yj depends on all nodes from each parent slate Yi (dense connections).
    Coefficients are scaled by 1/sqrt(num_parents) to prevent variance explosion.
    All functions are bounded (tanh, sin, cos) for numerical stability.
    """
    
    def __init__(self, dag_adjacency, sigma=1.0, correlation=0.5, function_seed=None):
        """
        Args:
            dag_adjacency: (6, 6) binary adjacency matrix, A[i,j] = 1 means slate i -> slate j
            sigma: Standard deviation for noise terms (default: 1.0)
            correlation: Correlation parameter for features within each slate (default: 0.5)
            function_seed: Seed to generate deterministic functional forms for this DAG
        """
        self.dag = np.asarray(dag_adjacency)
        self.sigma = sigma
        self.correlation = correlation
        self.n_slates = 6
        self.slate_size = 5
        
        if self.dag.shape != (self.n_slates, self.n_slates):
            raise ValueError(f"DAG adjacency must be ({self.n_slates}, {self.n_slates})")
        
        self._generate_functions(function_seed)
    
    def _generate_functions(self, seed):
        """
        Pre-generate random functional forms for each edge in the DAG.
        These are fixed given the seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.slate_functions = {}

        for j in range(self.n_slates):
            parents = np.where(self.dag[:, j] == 1)[0]

            if len(parents) == 0:
                self.slate_functions[j] = None  # root slate, no parents
                continue

            scale_factor = 1.0 / np.sqrt(len(parents))

            node_params = []
            for k in range(self.slate_size):
                parent_contributions = []
                for parent_idx in parents:
                    coeffs = np.random.uniform(0.4, 0.7, size=3) * scale_factor
                    func_types = np.random.randint(0, 6, size=3)
                    weights = np.random.uniform(-1, 1, size=self.slate_size)
                    parent_contributions.append({
                        'coeffs': coeffs,
                        'func_types': func_types,
                        'weights': weights
                    })
                node_params.append(parent_contributions)

            self.slate_functions[j] = node_params
    
    def _apply_complex_function(self, func_type, parent_values, weights):
        """
        Apply a complex nonlinear function to parent values.
        
        Args:
            func_type: Integer 0-5 selecting which function to use
            parent_values: (n, 5) array of parent slate values
            weights: (5,) array of mixing weights
        
        Returns:
            (n,) array of transformed values
        """
        weighted_sum = np.dot(parent_values, weights)

        if func_type == 0:
            return np.tanh(np.sum(parent_values**2, axis=1))
        elif func_type == 1:
            w2 = np.roll(weights, 2)
            return np.sin(weighted_sum) * np.cos(np.dot(parent_values, w2))
        elif func_type == 2:
            return np.tanh(parent_values[:, 0] * parent_values[:, 2])
        elif func_type == 3:
            return np.sin(parent_values[:, 1])**2 * np.tanh(parent_values[:, 3])
        elif func_type == 4:
            return np.tanh(weighted_sum**2)
        else:  # func_type == 5
            return np.cos(parent_values[:, 0]) * np.tanh(parent_values[:, 2] * parent_values[:, 4])
    
    def _generate_slate(self, X, slate_idx, n):
        """
        Generate values for a slate given its parents.
        
        Args:
            X: Current data matrix (n, 30)
            slate_idx: Index of slate to generate (0-5)
            n: Number of samples
        
        Returns:
            (n, 5) array of generated slate values
        """
        parents = np.where(self.dag[:, slate_idx] == 1)[0]
        node_params = self.slate_functions[slate_idx]

        correlation_matrix = np.eye(self.slate_size)
        for i in range(self.slate_size):
            for j in range(self.slate_size):
                if i != j:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = self.correlation ** distance
        
        mean = np.zeros(self.slate_size)
        epsilon = np.random.multivariate_normal(mean, correlation_matrix * self.sigma**2, size=n)
        
        slate_values = np.zeros((n, self.slate_size))

        for k in range(self.slate_size):
            total_contribution = 0.0

            for parent_idx, parent_contrib in zip(parents, node_params[k]):
                parent_start = parent_idx * self.slate_size
                parent_end = (parent_idx + 1) * self.slate_size
                parent_values = X[:, parent_start:parent_end]

                for i in range(3):
                    coeff = parent_contrib['coeffs'][i]
                    func_type = parent_contrib['func_types'][i]
                    weights = parent_contrib['weights']
                    
                    total_contribution += coeff * self._apply_complex_function(
                        func_type, parent_values, weights
                    )
            
            slate_values[:, k] = total_contribution + epsilon[:, k]
        
        return slate_values
    
    def sample(self, n):
        """
        Generate n samples from the random DAG.
        
        Args:
            n: Number of samples to generate
        
        Returns:
            X: (n, 30) array of samples
        """
        X = np.zeros((n, 30))
        
        for slate_idx in range(self.n_slates):
            parents = np.where(self.dag[:, slate_idx] == 1)[0]
            
            slate_start = slate_idx * self.slate_size
            slate_end = (slate_idx + 1) * self.slate_size
            
            if len(parents) == 0:
                # Root slate: sample from correlated normal.
                correlation_matrix = np.eye(self.slate_size)
                for i in range(self.slate_size):
                    for j in range(self.slate_size):
                        if i != j:
                            distance = abs(i - j)
                            correlation_matrix[i, j] = self.correlation ** distance
                
                mean = np.zeros(self.slate_size)
                X[:, slate_start:slate_end] = np.random.multivariate_normal(
                    mean, correlation_matrix * self.sigma**2, size=n
                )
            else:
                X[:, slate_start:slate_end] = self._generate_slate(X, slate_idx, n)
        
        return X


def compute_conditional_expectation_random(Y1_val, dag_adjacency, sigma=1.0, correlation=0.5, 
                                          function_seed=None, n_monte_carlo=100000):
    """
    Compute E[Y2, Y3, Y4, Y5, Y6 | Y1] for a random DAG via Monte Carlo simulation.
    
    Args:
        Y1_val: (5,) array-like, values to condition Y1 on
        dag_adjacency: (6, 6) binary adjacency matrix defining the DAG structure
        sigma: Standard deviation for noise terms (default: 1.0)
        correlation: Correlation parameter for features within each slate (default: 0.5)
        function_seed: Seed for generating functional forms (must match sampler)
        n_monte_carlo: Number of Monte Carlo samples for expectation estimation (default: 100000)
    
    Returns:
        Dictionary with conditional means of Y2-Y6 (5-dimensional vectors each) given Y1
    
    Note: We propagate uncertainty through the DAG by sampling according to the graph structure.
    """
    Y1_val = np.asarray(Y1_val)
    if Y1_val.shape[0] != 5:
        raise ValueError("Y1_val must be a 5-dimensional vector")
    
    dag = np.asarray(dag_adjacency)
    n_slates = 6
    slate_size = 5
    
    sampler = TrueSampler_random(dag, sigma, correlation, function_seed)

    X_samples = np.zeros((n_monte_carlo, 30))
    X_samples[:, 0:5] = Y1_val

    for slate_idx in range(1, n_slates):
        parents = np.where(dag[:, slate_idx] == 1)[0]

        slate_start = slate_idx * slate_size
        slate_end = (slate_idx + 1) * slate_size

        if len(parents) == 0:
            # Root slate (unlikely for slates 1-5, but handle gracefully).
            correlation_matrix = np.eye(slate_size)
            for i in range(slate_size):
                for j in range(slate_size):
                    if i != j:
                        distance = abs(i - j)
                        correlation_matrix[i, j] = correlation ** distance
            
            mean = np.zeros(slate_size)
            X_samples[:, slate_start:slate_end] = np.random.multivariate_normal(
                mean, correlation_matrix * sigma**2, size=n_monte_carlo
            )
        else:
            X_samples[:, slate_start:slate_end] = sampler._generate_slate(
                X_samples, slate_idx, n_monte_carlo
            )

    Y2_mean = np.mean(X_samples[:, 5:10], axis=0)
    Y3_mean = np.mean(X_samples[:, 10:15], axis=0)
    Y4_mean = np.mean(X_samples[:, 15:20], axis=0)
    Y5_mean = np.mean(X_samples[:, 20:25], axis=0)
    Y6_mean = np.mean(X_samples[:, 25:30], axis=0)
    
    return {
        'Y2_mean': Y2_mean,
        'Y3_mean': Y3_mean,
        'Y4_mean': Y4_mean,
        'Y5_mean': Y5_mean,
        'Y6_mean': Y6_mean,
        'Y2_std': np.std(X_samples[:, 5:10], axis=0),
        'Y3_std': np.std(X_samples[:, 10:15], axis=0),
        'Y4_std': np.std(X_samples[:, 15:20], axis=0),
        'Y5_std': np.std(X_samples[:, 20:25], axis=0),
        'Y6_std': np.std(X_samples[:, 25:30], axis=0),
        'indices': list(range(5, 30)),  # X5-X29 (all Y2-Y6)
        'names': [f'X{i}' for i in range(5, 30)]
    }