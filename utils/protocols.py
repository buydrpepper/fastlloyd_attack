from copy import copy
import numpy as np
from tqdm import tqdm

from parties import MaskedClient, UnmaskedClient, Server
from data_io import to_fixed
from configs import Params
from utils import set_seed


def mpi_proto(value_lists, params: Params, method="masked"):
    """Implements the MPI-based protocol for federated clustering.
    
    This protocol uses Message Passing Interface (MPI) for distributed computation,
    enabling communication between multiple processes representing different clients
    and a central server. It supports both masked (privacy-preserving) and unmasked
    operations.
    
    The protocol follows these steps in each iteration:
    1. Clients compute local statistics (totals and counts)
    2. These statistics are gathered at the server
    3. Server processes the aggregated statistics (possibly adding DP noise)
    4. Results are broadcast back to all clients
    5. Clients update their local centroids
    
    Args:
        value_lists (list): List of numpy arrays, where each array contains the data
                           points for one client
        params (Params): Configuration parameters for the clustering algorithm
        method (str, optional): Either "masked" for privacy-preserving computation
                              or "unmasked" for standard computation. Defaults to "masked"
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final cluster centroids after all iterations
            - int: Always 0 for MPI protocol (kept for compatibility with local_proto)
            
    Note:
        - The protocol uses the `data_io.comm` module for MPI operations
        - Progress is shown only on the server process using tqdm
        - Communication statistics are tracked and printed at the end
    """
    from data_io.comm import comm, fail_together
    set_seed(params.seed)
    comm.reset_comm_stats()
    comm.set_delay(params.delay)
    server_process = (comm.rank == comm.root)

    def initialize_server():
        """Initialize the server process with given parameters."""
        return Server(params)

    def initialize_client():
        """Initialize a client (masked or unmasked) with appropriate data."""
        cls = MaskedClient if method == "masked" else UnmaskedClient
        return cls(comm.rank - 1, value_lists[comm.rank - 1], params)

    if server_process:
        server = fail_together(initialize_server, "Server Initialization Failure")
    else:
        client = fail_together(initialize_client, "Client Initialization Failure")
    pbar = tqdm(range(params.iters)) if server_process else range(params.iters)

    for i in pbar:
        params.update_maxdist(i)
        if not server_process:
            # Client-side computation
            total, count, _ = client.step(params)

            # Pack statistics into a single array for efficient communication
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.gather_delay(total_count, root=comm.root)

            # Receive and unpack aggregated statistics from server
            total_count = comm.bcast_delay(None, root=comm.root)

            total, count = np.split(total_count, [params.k * params.dim])
            total = total.reshape((params.k, params.dim))
            count = count.reshape(params.k)
            client.update(total, count)

        if server_process:
            # Server-side computation
            total_counts = comm.gather_delay(None, root=comm.root)
            total_counts = [np.split(tc, [params.k * params.dim]) for tc in total_counts[1:]]
            totals, counts = zip(*[(total.reshape((params.k, params.dim)), count.reshape(params.k))
                                   for total, count in total_counts])
            total, count = server.step(totals, counts, params)

            # Pack and broadcast updated statistics
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.bcast_delay(total_count, root=comm.root)

        # Synchronize centroids across all processes
        centroids = comm.bcast(client.centroids if not server_process else None, root=1)

    comm.print_comm_stats()
    return [to_fixed(centroids)], 0



def local_proto(value_lists, params: Params, method="masked"):
    """Implements the local protocol for federated clustering.
    
    This protocol simulates federated clustering in a single process, useful for
    testing and development. It maintains separate client and server instances
    in memory and simulates their interaction. Like the MPI protocol, it supports
    both masked and unmasked computation.
    
    The protocol follows these steps in each iteration:
    1. Each client computes local statistics
    2. The server aggregates these statistics
    3. Clients update their centroids using the aggregated statistics
    4. Progress is tracked through centroid movement
    
    The implementation also tracks the number of unassigned points (points too
    far from any centroid)
    
    Args:
        value_lists (list): List of numpy arrays, where each array contains the data
                           points for one client
        params (Params): Configuration parameters for the clustering algorithm
        method (str, optional): Either "masked" for privacy-preserving computation
                              or "unmasked" for standard computation. Defaults to "masked"
    
    Returns:
        tuple: A tuple containing:
            - list: a list containing the centroid history in fixed point
            - int: Number of points not assigned to any cluster in the final iteration
            - np.ndarray: the centroids in the final iteration
            - np.ndarray: the return value of lloyd's algorithm applied to the data
            - list: a list containing tuples (total, count), of types list(np.ndarray (k, d))

            
    Note:
        - Progress bar shows the Euclidean norm of centroid movement between iterations
        - All clients maintain identical centroids due to synchronized updates
    """
    set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    centroid_history = [centroids]
    server = Server(params)
    pbar = tqdm(range(params.iters))
    unassigned_last_iter = 0
    total_and_count_history = [([],[])]

    for i in pbar:
        params.update_maxdist(i)
        # Collect statistics from all clients
        totals = []
        counts = []
        unassigneds = []
        for client in clients:
            total, count, unassigned = client.step(params)
            totals.append(total)
            counts.append(count)
            unassigneds.append(unassigned)
        unassigned_last_iter = sum(unassigneds)
        total_and_count_history[len(total_and_count_history)-1]=(totals,counts)
        # Server processes aggregated statistics
        total, count = server.step(totals, counts, params)
        total_and_count_history.append((totals,counts))

        # Update all clients
        for client in clients:
            client.update(total, count)

        # Track progress through centroid movement
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
        centroid_history.append(centroids)

    return list(map(to_fixed, centroid_history)), unassigned_last_iter, to_fixed(centroid_history[-1]), get_lloyd_centroid(value_lists, params), total_and_count_history


def get_lloyd_centroid(value_lists, paramref: Params, method="unmasked"):

    params=copy(paramref)
    setattr(params, 'dp', "none")
    setattr(params, 'method', "none")
    setattr(params, 'post', "none")
    setattr(params, 'alpha', 0)

    #Don't set seed
    #set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    centroid_history = [centroids]
    server = Server(params)
    pbar = tqdm(range(params.iters))
    unassigned_last_iter = 0

    for i in pbar:
        params.update_maxdist(i)
        # Collect statistics from all clients
        totals = []
        counts = []
        unassigneds = []
        for client in clients:
            total, count, unassigned = client.step(params)
            totals.append(total)
            counts.append(count)
            unassigneds.append(unassigned)
        unassigned_last_iter = sum(unassigneds)

        # Server processes aggregated statistics
        total, count = server.step(totals, counts, params)

        # Update all clients
        for client in clients:
            client.update(total, count)

        # Track progress through centroid movement
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
        centroid_history.append(centroids)
    return to_fixed(centroids)


def custom_proto(value_lists, params: Params, dp= "", conv=False):
    """Projects the final guess onto the median line"""
    set_seed(params.seed)
    cls = UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    centroid_history = [centroids] #necessary, since post processing is done
    total_history=[]
    count_history=[]
    server = Server(params)
    pbar = tqdm(range(params.iters))
    unassigned_last_iter = 0

    err = 0
    for i in pbar:
        params.update_maxdist(i)
        # Collect statistics from all clients
        totals = []
        counts = []
        unassigneds = []
        for client in clients:
            total, count, unassigned = client.step(params)
            totals.append(total)
            counts.append(count)
            unassigneds.append(unassigned)
        unassigned_last_iter = sum(unassigneds)

        # Server processes aggregated statistics
        total, count = server.step(totals, counts, params)

        # Update all clients
        for client in clients:
            client.update(total, count)

        # Track progress through centroid movement
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
        centroid_history.append(centroids)
        total_history.append(total)
        count_history.append(count)

    # with open("error" + str(params.eps), "a") as f:
    #      f.write(str(err) + "\n")
    if conv:
        if False and ((params.eps == 0.1 and err >= 0.675339*0.75) or
            (params.eps == 0.25 and err >= 0.517732*0.75) or
            (params.eps == 0.5 and err >= 0.403405*0.75) or
            (params.eps == 0.75 and err >= 0.199732*0.75) or
            (params.eps == 1 and err >= 0.153413*0.75)):

            return list(map(to_fixed, centroid_history)), unassigned_last_iter, to_fixed(centroid_history[-1]), get_lloyd_centroid(value_lists, params)

        if False:
            return list(map(to_fixed, centroid_history)), unassigned_last_iter, to_fixed(centroid_history[-1]), get_lloyd_centroid(value_lists, params)


    if "project" in dp:
        if params.iters < 2:
            return list(map(to_fixed, centroid_history)), unassigned_last_iter, to_fixed(centroid_history[-1]), get_lloyd_centroid(value_lists, params)

        history_arr = np.array(centroid_history) 
        if "last" in dp:
            history_arr = history_arr[-3:]

        directionlst = []
        meanlst = []

        for i in range(params.k):
            cluster_path = history_arr[:, i, :]

            data_mean = cluster_path.mean(axis=0) #dimension (dim)
            centered_data = cluster_path - data_mean

            _, _, Vh = np.linalg.svd(centered_data)
            direction_vector = Vh[0]
            directionlst.append(direction_vector)
            meanlst.append(data_mean)

        projected_centroids = np.zeros((params.k,params.dim))

        for i in range(params.k):
            a = meanlst[i]
            u = centroids[i]-a
            v = directionlst[i]
            t = np.dot(u,v)/np.dot(v,v)

            projected_centroids[i] = a + t*v

        return list(map(to_fixed, centroid_history)), unassigned_last_iter,to_fixed(projected_centroids), get_lloyd_centroid(value_lists, params)


    elif "average" in dp:
        return list(map(to_fixed, centroid_history)), unassigned_last_iter, to_fixed((centroids+centroid_history[-2])/2) if params.iters>0 else to_fixed(centroid_history[-1]), get_lloyd_centroid(value_lists, params)
    else:
        sys.exit()

def no_final_noise(value_lists, params: Params, method="masked"):
    """Same as local, no noise on final iterate"""
    set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    centroid_history = [centroids]
    server = Server(params)
    pbar = tqdm(range(params.iters))
    unassigned_last_iter = 0

    for i in pbar:
        params.update_maxdist(i)
        # Collect statistics from all clients
        totals = []
        counts = []
        unassigneds = []
        for client in clients:
            total, count, unassigned = client.step(params)
            totals.append(total)
            counts.append(count)
            unassigneds.append(unassigned)
        unassigned_last_iter = sum(unassigneds)

        # Server processes aggregated statistics
        if i == params.iters - 1:
            tmpserver = copy(server)
            setattr(tmpserver.params, 'dp', "none")
            total,count = tmpserver.step(totals,counts,params) #params doesn't matter here
        else:
            total, count = server.step(totals, counts, params)

        #NOTE: Server is no longer used after the final iteration.

        # Update all clients
        for client in clients:
            client.update(total, count)

        # Track progress through centroid movement
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
        centroid_history.append(centroids)

    return list(map(to_fixed, centroid_history)), unassigned_last_iter, to_fixed(centroid_history[-1]), get_lloyd_centroid(value_lists, params)

