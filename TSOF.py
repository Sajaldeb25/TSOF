import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import depth_first_order
import networkx as nx
from sklearn.metrics import confusion_matrix

def condition_one(g, v, density):
    """
    Implementation of Condition_one function used in offline training.
    This function extracts data cloud centers based on the graph structure.
    
    Parameters:
    -----------
    g : array-like
        Graph connections array
    v : array-like
        DFS search order
    density : array-like
        Density values
    
    Returns:
    --------
    data_cloud_cen : list
        Indices of data cloud centers
    """
    # Simple implementation to find local maxima in the graph
    data_cloud_cen = [v[0]]  # Always include the root node
    
    # Map to track which nodes have been visited
    visited = {}
    for node in v:
        visited[node] = False
    visited[v[0]] = True
    
    # Find local maxima in the tree
    for i in range(1, len(v)):
        node = v[i]
        # Find parent node in the graph
        for j in range(len(g)):
            if g[j, 1] == node:
                parent = g[j, 0]
                break
        
        # If density at this node is higher than its parent, it's a local maximum
        if density[node-1] > density[parent-1]:
            data_cloud_cen.append(node)
        
        visited[node] = True
    
    return data_cloud_cen

def offline_training_euclidean(data, delta, gran_level):
    """
    Offline training for the TSOF classifier using Euclidean distance.
    
    Parameters:
    -----------
    data : array-like
        Training data
    delta : float
        Variance of data
    gran_level : int
        Granulation level parameter
    
    Returns:
    --------
    centre3 : array-like
        Centers of data clouds
    mnumber : array-like
        Number of members in each data cloud
    averdist : float
        Average distance threshold
    """
    L, W = data.shape
    
    # Calculate pairwise distances
    dist00 = pdist(data, 'euclidean')**2
    dist0 = squareform(dist00)
    
    # Calculate average distance
    dist00_sorted = np.sort(dist00)
    for i in range(gran_level):
        dist00_sorted = dist00_sorted[dist00_sorted <= np.mean(dist00_sorted)]
    averdist = np.mean(dist00_sorted)
    
    # Find unique data points
    unique_rows = {}
    J = []
    K = []
    for i, row in enumerate(data):
        row_tuple = tuple(row)
        if row_tuple not in unique_rows:
            unique_rows[row_tuple] = len(J)
            J.append(i)
        K.append(unique_rows[row_tuple])
    
    UD = data[J]
    
    # Count occurrences of each unique row
    F = np.zeros(len(J))
    for k in K:
        F[k] += 1
    
    LU = len(UD)
    
    # Calculate density
    density = np.sum(dist0, axis=1) / np.sum(np.sum(dist0))
    density = F / density[J]
    dist = dist0[np.ix_(J, J)]
    
    # Find point with maximum density
    pos = np.argmax(density)
    seq = list(range(LU))
    seq.remove(pos)
    
    # Build ranking
    rank = np.zeros(LU, dtype=int)
    rank[0] = pos
    
    # Build graph for MST
    g = np.zeros((LU, 2), dtype=int)
    g[0] = [pos, pos]
    
    for i in range(1, LU):
        pos_list = rank[rank >= 0]
        aa = dist[pos_list[0], seq]
        for pos_idx in range(1, len(pos_list)):
            aa = np.minimum(aa, dist[pos_list[pos_idx], seq])
        
        if i == 1:
            pos0 = np.argmin(aa)
            g[i] = [pos, seq[pos0]]
            pos = seq[pos0]
            rank[i] = pos
            seq.remove(pos)
        else:
            pos1 = np.argmin(aa)
            x = seq[pos1]
            g[i] = [rank[np.argmin(dist[rank[rank > 0], x])], x]
            pos = x
            rank[i] = pos
            seq.remove(pos)
    
    # Create graph and perform DFS
    G = nx.DiGraph()
    for i in range(len(g)):
        G.add_edge(g[i, 0], g[i, 1])
    
    v = list(nx.dfs_preorder_nodes(G, g[0, 0]))
    
    # Get data cloud centers
    data_cloud_cen = condition_one(g, v, density)
    
    # Extract centers
    centre0 = UD[data_cloud_cen]
    
    # Assign data points to nearest centers
    nc = centre0.shape[0]
    dist3 = cdist(centre0, data, 'euclidean')**2
    seq4 = np.argmin(dist3, axis=0)
    
    # Calculate centers and member counts
    centre1 = np.zeros((nc, W))
    mnumber = np.zeros(nc)
    miu = np.mean(data, axis=0)
    cenden = np.zeros(nc)
    
    for i in range(nc):
        seq5 = np.where(seq4 == i)[0]
        mnumber[i] = len(seq5)
        centre1[i] = np.mean(data[seq5], axis=0)
        cenden[i] = mnumber[i] / (1 + np.sum((centre1[i] - miu)**2) / delta)
    
    # Calculate distances between centers
    dist4 = pdist(centre1, 'euclidean')**2
    dist5 = squareform(dist4)
    
    # Filter centers by density
    seqme2 = np.zeros((nc, nc))
    seqme2[dist5 <= averdist] = 1
    cendenmex = seqme2 * np.tile(cenden, (nc, 1))
    seq6 = np.where(np.abs(np.max(cendenmex, axis=1) - cenden) == 0)[0]
    
    centre2 = centre1[seq6]
    
    # Final assignment
    nc = centre2.shape[0]
    dist6 = cdist(centre2, data, 'euclidean')**2
    seq7 = np.argmin(dist6, axis=0)
    
    centre3 = np.zeros((nc, W))
    mnumber = np.zeros(nc)
    
    for i in range(nc):
        seq8 = np.where(seq7 == i)[0]
        mnumber[i] = len(seq8)
        centre3[i] = np.mean(data[seq8], axis=0)
    
    return centre3, mnumber, averdist

def evolving_training_euclidean(data_point, mu, centre, member, delta, threshold):
    """
    Evolving training for the TSOF classifier using Euclidean distance.
    
    Parameters:
    -----------
    data_point : array-like
        New data point for online learning
    mu : array-like
        Mean of the data
    centre : array-like
        Centers of data clouds
    member : array-like
        Number of members in each data cloud
    delta : float
        Variance of data
    threshold : float
        Distance threshold
    
    Returns:
    --------
    centre : array-like
        Updated centers of data clouds
    member : array-like
        Updated number of members in each data cloud
    """
    # Calculate distances
    dist1 = cdist(centre.reshape(-1, len(mu)), mu.reshape(1, -1), 'euclidean')**2
    dist2 = np.sum((data_point - mu)**2)
    
    if dist2 > np.max(dist1) or dist2 < np.min(dist1):
        # Add new center
        centre = np.vstack((centre, data_point))
        member = np.append(member, 1)
    else:
        # Find closest center
        dist3 = cdist(data_point.reshape(1, -1), centre, 'euclidean')**2
        pos3 = np.argmin(dist3)
        
        if dist3[0, pos3] > threshold:
            # Add new center
            centre = np.vstack((centre, data_point))
            member = np.append(member, 1)
        else:
            # Update existing center
            centre[pos3] = (member[pos3] * centre[pos3] + data_point) / (member[pos3] + 1)
            member[pos3] += 1
    
    return centre, member

def offline_training_mahalanobis(data, cov_data, gran_level):
    """
    Offline training for the TSOF classifier using Mahalanobis distance.
    
    Parameters:
    -----------
    data : array-like
        Training data
    cov_data : array-like
        Covariance matrix of the data
    gran_level : int
        Granulation level parameter
    
    Returns:
    --------
    centre3 : array-like
        Centers of data clouds
    mnumber : array-like
        Number of members in each data cloud
    averdist : float
        Average distance threshold
    """
    L, W = data.shape
    
    # Calculate pairwise distances using Mahalanobis distance
    try:
        # Handle potential numerical issues with covariance matrix
        inv_cov = np.linalg.inv(cov_data)
        dist00 = []
        for i in range(L):
            for j in range(i+1, L):
                diff = data[i] - data[j]
                dist00.append(np.dot(np.dot(diff, inv_cov), diff))
        dist00 = np.array(dist00)
    except:
        # Fallback to Euclidean if Mahalanobis fails
        dist00 = pdist(data, 'euclidean')**2
    
    dist0 = squareform(dist00)
    
    # Calculate average distance
    dist00_sorted = np.sort(dist00)
    for i in range(gran_level):
        if len(dist00_sorted) > 0:
            dist00_sorted = dist00_sorted[dist00_sorted <= np.mean(dist00_sorted)]
    averdist = np.mean(dist00_sorted) if len(dist00_sorted) > 0 else 0
    
    # Find unique data points
    unique_rows = {}
    J = []
    K = []
    for i, row in enumerate(data):
        row_tuple = tuple(row)
        if row_tuple not in unique_rows:
            unique_rows[row_tuple] = len(J)
            J.append(i)
        K.append(unique_rows[row_tuple])
    
    UD = data[J]
    
    # Count occurrences of each unique row
    F = np.zeros(len(J))
    for k in K:
        F[k] += 1
    
    LU = len(UD)
    
    # Calculate density
    density = np.sum(dist0, axis=1) / np.sum(np.sum(dist0))
    density = F / density[J]
    dist = dist0[np.ix_(J, J)]
    
    # Find point with maximum density
    pos = np.argmax(density)
    seq = list(range(LU))
    seq.remove(pos)
    
    # Build ranking
    rank = np.zeros(LU, dtype=int)
    rank[0] = pos
    
    for i in range(1, LU):
        aa = dist[pos, seq]
        pos0 = np.argmin(aa)
        pos = seq[pos0]
        rank[i] = pos
        seq.remove(pos)
    
    data2 = UD[rank]
    data2den = density[rank]
    
    # Find peaks in density gradient
    gradient = np.zeros((2, LU-2))
    gradient[0] = data2den[:-2] - data2den[1:-1]
    gradient[1] = data2den[1:-1] - data2den[2:]
    
    seq2 = list(range(1, LU-1))
    seq1 = np.where((gradient[0] < 0) & (gradient[1] > 0))[0]
    
    if gradient[1, -1] < 0:
        seq3 = [0] + [seq2[i] for i in seq1] + [LU-1]
    else:
        seq3 = [0] + [seq2[i] for i in seq1]
    
    centre0 = data2[seq3]
    
    # Assign data points to nearest centers
    nc = centre0.shape[0]
    try:
        dist3 = cdist(centre0, data, 'mahalanobis', VI=inv_cov)**2
    except:
        dist3 = cdist(centre0, data, 'euclidean')**2
    
    seq4 = np.argmin(dist3, axis=0)
    
    # Calculate centers and member counts
    centre1 = np.zeros((nc, W))
    mnumber = np.zeros(nc)
    miu = np.mean(data, axis=0)
    cenden = np.zeros(nc)
    
    for i in range(nc):
        seq5 = np.where(seq4 == i)[0]
        mnumber[i] = len(seq5)
        centre1[i] = np.mean(data[seq5], axis=0)
        
        # Calculate density
        diff = centre1[i] - miu
        try:
            cenden[i] = mnumber[i] / (1 + np.dot(np.dot(diff, inv_cov), diff) / W)
        except:
            cenden[i] = mnumber[i] / (1 + np.sum((diff)**2) / W)
    
    # Calculate distances between centers
    try:
        dist4 = pdist(centre1, 'mahalanobis', VI=inv_cov)**2
    except:
        dist4 = pdist(centre1, 'euclidean')**2
    
    dist5 = squareform(dist4)
    
    # Filter centers by density
    seqme2 = np.zeros((nc, nc))
    seqme2[dist5 <= averdist] = 1
    cendenmex = seqme2 * np.tile(cenden, (nc, 1))
    seq6 = np.where(np.abs(np.max(cendenmex, axis=1) - cenden) == 0)[0]
    
    centre2 = centre1[seq6]
    
    # Final assignment
    nc = centre2.shape[0]
    try:
        dist6 = cdist(centre2, data, 'mahalanobis', VI=inv_cov)**2
    except:
        dist6 = cdist(centre2, data, 'euclidean')**2
    
    seq7 = np.argmin(dist6, axis=0)
    
    centre3 = np.zeros((nc, W))
    mnumber = np.zeros(nc)
    
    for i in range(nc):
        seq8 = np.where(seq7 == i)[0]
        mnumber[i] = len(seq8)
        centre3[i] = np.mean(data[seq8], axis=0)
    
    return centre3, mnumber, averdist

def evolving_training_mahalanobis(data_point, mu, centre, member, cov_data, threshold):
    """
    Evolving training for the TSOF classifier using Mahalanobis distance.
    
    Parameters:
    -----------
    data_point : array-like
        New data point for online learning
    mu : array-like
        Mean of the data
    centre : array-like
        Centers of data clouds
    member : array-like
        Number of members in each data cloud
    cov_data : array-like
        Covariance matrix
    threshold : float
        Distance threshold
    
    Returns:
    --------
    centre : array-like
        Updated centers of data clouds
    member : array-like
        Updated number of members in each data cloud
    """
    try:
        inv_cov = np.linalg.inv(cov_data)
        dist1 = cdist(centre, mu.reshape(1, -1), 'mahalanobis', VI=inv_cov)**2
        diff = data_point - mu
        dist2 = np.dot(np.dot(diff, inv_cov), diff)
    except:
        dist1 = cdist(centre, mu.reshape(1, -1), 'euclidean')**2
        dist2 = np.sum((data_point - mu)**2)
    
    if dist2 > np.max(dist1) or dist2 < np.min(dist1):
        # Add new center
        centre = np.vstack((centre, data_point))
        member = np.append(member, 1)
    else:
        # Find closest center
        try:
            dist3 = cdist(data_point.reshape(1, -1), centre, 'mahalanobis', VI=inv_cov)**2
        except:
            dist3 = cdist(data_point.reshape(1, -1), centre, 'euclidean')**2
        
        pos3 = np.argmin(dist3)
        
        if dist3[0, pos3] > threshold:
            # Add new center
            centre = np.vstack((centre, data_point))
            member = np.append(member, 1)
        else:
            # Update existing center
            centre[pos3] = (member[pos3] * centre[pos3] + data_point) / (member[pos3] + 1)
            member[pos3] += 1
    
    return centre, member

def sof_classifier(input_data, gran_level, mode, distance_type):
    """
    Main function for the Tree-based Self-Organizing Fuzzy (T-SOF) classifier.
    
    Parameters:
    -----------
    input_data : dict
        Input data and parameters for the classifier
    gran_level : int
        Granulation level parameter
    mode : str
        Mode of operation ('OfflineTraining', 'EvolvingTraining', or 'Validation')
    distance_type : str
        Distance metric ('Euclidean', 'Mahalanobis', or 'Cosine')
    
    Returns:
    --------
    output : dict
        Results of the classifier operation
    """
    output = {}
    
    # Offline Training
    if mode == 'OfflineTraining':
        data_train = input_data['TrainingData']
        label_train = input_data['TrainingLabel']
        seq = np.unique(label_train)
        data_train1 = {}
        N = len(seq)
        
        # Apply Cosine normalization if selected
        if distance_type == 'Cosine':
            norms = np.sqrt(np.sum(data_train**2, axis=1)).reshape(-1, 1)
            data_train = data_train / norms
            distance_type = 'Euclidean'
        
        if distance_type == 'Euclidean':
            centre = {}
            member = {}
            averdist = {}
            delta = np.zeros(N)
            
            for i in range(N):
                data_train1[i] = data_train[label_train == seq[i]]
                delta[i] = np.mean(np.sum(data_train1[i]**2, axis=1)) - np.sum(np.mean(data_train1[i], axis=0)**2)
                centre[i], member[i], averdist[i] = offline_training_euclidean(data_train1[i], delta[i], gran_level)
            
            L = np.zeros(N)
            mu = {}
            XX = np.zeros(N)
            ratio = np.zeros(N)
            
            for i in range(N):
                mu[i] = np.mean(data_train1[i], axis=0)
                L[i] = len(data_train1[i])
                XX[i] = np.mean(np.sum(data_train1[i]**2, axis=1))
                ratio[i] = averdist[i] / (2 * (XX[i] - np.sum(mu[i]**2)))
            
            trained_classifier = {
                'seq': seq,
                'ratio': ratio,
                'miu': mu,
                'XX': XX,
                'L': L,
                'centre': centre,
                'Member': member,
                'averdist': averdist,
                'NoC': N,
                'delta': delta
            }
        
        elif distance_type == 'Mahalanobis':
            centre = {}
            member = {}
            averdist = {}
            cov_data = {}
            
            for i in range(N):
                data_train1[i] = data_train[label_train == seq[i]]
                cov_data[i] = np.cov(data_train1[i], rowvar=False)
                centre[i], member[i], averdist[i] = offline_training_mahalanobis(data_train1[i], cov_data[i], gran_level)
            
            L = np.zeros(N)
            mu = {}
            XX = {}
            threshold = {}
            
            for i in range(N):
                mu[i] = np.mean(data_train1[i], axis=0)
                L[i] = len(data_train1[i])
                XX[i] = np.zeros((data_train1[i].shape[1], data_train1[i].shape[1]))
                
                for ii in range(int(L[i])):
                    XX[i] += np.outer(data_train1[i][ii], data_train1[i][ii])
                
                XX[i] /= L[i]
                threshold[i] = averdist[i]
            
            trained_classifier = {
                'seq': seq,
                'miu': mu,
                'XX': XX,
                'L': L,
                'centre': centre,
                'Member': member,
                'threshold': threshold,
                'NoC': N,
                'covMatrix': cov_data
            }
        
        output['TrainedClassifier'] = trained_classifier
    
    # Evolving Training
    elif mode == 'EvolvingTraining':
        data_train = input_data['TrainingData']
        label_train = input_data['TrainingLabel']
        trained_classifier = input_data['TrainedClassifier']
        
        # Apply Cosine normalization if selected
        if distance_type == 'Cosine':
            norms = np.sqrt(np.sum(data_train**2, axis=1)).reshape(-1, 1)
            data_train = data_train / norms
            distance_type = 'Euclidean'
        
        if distance_type == 'Euclidean':
            seq = trained_classifier['seq']
            ratio = trained_classifier['ratio']
            mu = trained_classifier['miu']
            XX = trained_classifier['XX']
            L = trained_classifier['L']
            centre = trained_classifier['centre']
            member = trained_classifier['Member']
            averdist = trained_classifier['averdist']
            delta = trained_classifier['delta']
            N = trained_classifier['NoC']
            
            data_train2 = {}
            for i in range(N):
                data_train2[i] = data_train[label_train == seq[i]]
            
            for i in range(N):
                for j in range(len(data_train2[i])):
                    L[i] += 1
                    XX[i] = XX[i] * (L[i] - 1) / L[i] + np.sum(data_train2[i][j]**2) / L[i]
                    mu[i] = mu[i] * (L[i] - 1) / L[i] + data_train2[i][j] / L[i]
                    delta[i] = XX[i] - np.sum(mu[i]**2)
                    threshold = 2 * delta[i] * ratio[i]
                    centre[i], member[i] = evolving_training_euclidean(data_train2[i][j], mu[i], centre[i], member[i], delta[i], threshold)
            
            trained_classifier['ratio'] = ratio
            trained_classifier['miu'] = mu
            trained_classifier['XX'] = XX
            trained_classifier['L'] = L
            trained_classifier['centre'] = centre
            trained_classifier['Member'] = member
            trained_classifier['averdist'] = averdist
            trained_classifier['NoC'] = N
            trained_classifier['delta'] = delta
        
        elif distance_type == 'Mahalanobis':
            seq = trained_classifier['seq']
            mu = trained_classifier['miu']
            XX = trained_classifier['XX']
            L = trained_classifier['L']
            centre = trained_classifier['centre']
            member = trained_classifier['Member']
            cov_data = trained_classifier['covMatrix']
            threshold = trained_classifier['threshold']
            N = trained_classifier['NoC']
            
            data_train2 = {}
            for i in range(N):
                data_train2[i] = data_train[label_train == seq[i]]
            
            for i in range(N):
                for j in range(len(data_train2[i])):
                    L[i] += 1
                    XX[i] = XX[i] * (L[i] - 1) / L[i] + np.outer(data_train2[i][j], data_train2[i][j]) / L[i]
                    mu[i] = mu[i] * (L[i] - 1) / L[i] + data_train2[i][j] / L[i]
                    cov_data[i] = L[i] / (L[i] - 1) * (XX[i] - np.outer(mu[i], mu[i]))
                    threshold1 = threshold[i]
                    centre[i], member[i] = evolving_training_mahalanobis(data_train2[i][j], mu[i], centre[i], member[i], cov_data[i], threshold1)
            
            trained_classifier['seq'] = seq
            trained_classifier['miu'] = mu
            trained_classifier['XX'] = XX
            trained_classifier['L'] = L
            trained_classifier['centre'] = centre
            trained_classifier['Member'] = member
            trained_classifier['threshold'] = threshold
            trained_classifier['NoC'] = N
            trained_classifier['covMatrix'] = cov_data
        
        output['TrainedClassifier'] = trained_classifier
    
    # Validation
    elif mode == 'Validation':
        trained_classifier = input_data['TrainedClassifier']
        seq = trained_classifier['seq']
        data_test = input_data['TestingData']
        label_test = input_data['TestingLabel']
        N = trained_classifier['NoC']
        
        # Apply Cosine normalization if selected
        if distance_type == 'Cosine':
            norms = np.sqrt(np.sum(data_test**2, axis=1)).reshape(-1, 1)
            data_test = data_test / norms
            distance_type = 'Euclidean'
        
        if distance_type == 'Euclidean':
            centre = trained_classifier['centre']
            dist = np.zeros((len(data_test), N))
            
            for i in range(N):
                dist[:, i] = np.min(cdist(data_test, centre[i], 'euclidean')**2, axis=1)
            
            label_est = np.argmin(dist, axis=1)
            label_est = seq[label_est]
        
        elif distance_type == 'Mahalanobis':
            cov_data = trained_classifier['covMatrix']
            centre = trained_classifier['centre']
            dist = np.zeros((len(data_test), N))
            
            for i in range(N):
                try:
                    dist[:, i] = np.min(cdist(data_test, centre[i], 'mahalanobis', VI=np.linalg.inv(cov_data[i]))**2, axis=1)
                except:
                    dist[:, i] = np.min(cdist(data_test, centre[i], 'euclidean')**2, axis=1)
            
            label_est = np.argmin(dist, axis=1)
            label_est = seq[label_est]
        
        output['TrainedClassifier'] = input_data['TrainedClassifier']
        output['ConfusionMatrix'] = confusion_matrix(label_test, label_est)
        output['EstimatedLabel'] = label_est
    
    return output
