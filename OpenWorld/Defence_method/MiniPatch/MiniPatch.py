import numpy as np

def get_pertubation(Dataset_name,CFmodel_name):
    # if Dataset_name == "DF":
    #     if CFmodel_name == "DF":
    #         noise=[]
    # noise=[78,-4,90,-7,160,3,26,-7,22,8,7,-8,53,0,80,1,109,6,173,-8]  #开销更大
    
    if Dataset_name == "sirinam95":
        noise=[34,6,9,6,26,6,17,6,2,6]
        
    elif Dataset_name == "rimmer100":
        noise=[18,6,25,6,40,6,12,6,3,5]
    else:
        raise ValueError("Unknown Dataset_name for MiniPatch perturbation.")
    noise= np.array(noise)
    return noise

def perturb_trace(traces, perturbations, highlight=False):
    """
    Perturb packet trace(s) according to the given perturbation(s).
    Not support multiple traces and multiple perturbations at the same time.

    Parameters
    ----------
    traces : array_like
        A 2-D numpy array [Length x 1] for a single trace or a 3-D numpy
        array [N x Length x 1] for N traces.
    perturbations : array_like
        A 1-D numpy array specifying a single perturbation or a 2-D numpy
        array specifying multiple perturbations.
    highlight : optional
        Highlight perturbations by setting the absolute value to 2.
    """
    if type(perturbations) == list:
        perturbations = np.array(perturbations)

    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if perturbations.ndim < 2:
        perturbations = np.array([perturbations])

    if traces.ndim < 3 or traces.shape[0] == 1:
        # Copy the trace n == len(perturbations) times so that we can 
        # create n new perturbed traces
        traces = np.tile(traces, [len(perturbations), 1, 1])
    else:
        # Copy the perturbation n == len(traces) times to 
        # create n new perturbed traces
        perturbations = np.tile(perturbations, [len(traces), 1])

    # Make sure to floor the members of perturbations as int types
    perturbations = perturbations.astype(int)

    for trace, perturbation in zip(traces, perturbations):
        # Split perturbation into an array of patches
        patches = np.split(perturbation, len(perturbation)//2)
        length = len(np.where(trace != 0)[0])

        # Align patch positions
        for patch in patches:
            x_pos, n_pkt = patch
            # Constraint 1: within the trace
            if x_pos > length:
                x_pos = length
            # Constraint 2: at burst tail with the same direction
            if x_pos < len(trace):
                while trace[x_pos, 0] * n_pkt < 0:
                    x_pos += 1
                    if x_pos >= len(trace):
                        break
            if x_pos < len(trace):
                while trace[x_pos, 0] * n_pkt > 0:
                    x_pos += 1
                    if x_pos >= len(trace):
                        break
            patch[0] = x_pos

        # Apply patches
        positions = []
        for patch in sorted(patches, key=lambda x: x[0], reverse=True):
            x_pos, n_pkt = patch
            direction = 1 if n_pkt > 0 else -1
            n_pkt = abs(n_pkt)
            # Constraint 3: at different positions
            if x_pos in positions:
                continue
            positions.append(x_pos)
            # Constraint 1: within the trace
            if x_pos + n_pkt >= len(trace):
                n_pkt = len(trace) - x_pos
            if n_pkt == 0:
                continue
            # Constraint 2: with the same direction
            if x_pos < length:
                assert direction * trace[x_pos-1, 0] > 0

            # At each trace's position x_pos, insert a patch of n packets
            trace[x_pos+n_pkt:, 0] = trace[x_pos:-n_pkt, 0]
            # Outgoing
            if direction > 0:
                if highlight:
                    trace[x_pos:x_pos+n_pkt, 0] = 2.
                else:
                    trace[x_pos:x_pos+n_pkt, 0] = 1.
            # Incoming
            else:
                if highlight:
                    trace[x_pos:x_pos+n_pkt, 0] = -2.
                else:
                    trace[x_pos:x_pos+n_pkt, 0] = -1.

    # Return shape: [N * Length * 1]
    return traces


def MiniPatch_pertubation(Dataset_name, CFmodel_name,data):
    data_copy = data.copy() 
    noise=get_pertubation(Dataset_name, CFmodel_name)
    perturbed_data = perturb_trace(data_copy, noise)
    return perturbed_data