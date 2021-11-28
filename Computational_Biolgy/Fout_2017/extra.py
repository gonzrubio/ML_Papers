def getLigData(lig, IND, device="cpu"):
    """Return x and Y for a ligand dataset.

    Parameters
    ----------
    lig : dict
        They keys (lig.keys()) are properties of a ligand.
    IND : list
        The indices of the molecules (the batch).
    device : str, cpu by default
        Where to load the data. Move X and Y to device.
    Returns
    -------
    II, JJ : Torch.LongTensor of shape [num_edges_in_batch]
        The edges of the graph. II is the source and JJ is the target node.
    XN, XE : Torch.FloatTensor of shape [1, num_node_feats, num_atoms_in_batch]
              and [1, num_edge_feats, num_edges_in_batch]
        The node/edge feature matrix of all the ligands in the batch.
    NL : Torch.LongTensor of shape [len(IND)]
        Number of atoms in each ligand present in the batch.
    score : Torch.FloatTensor of shape [len(IND), 1]
        The true binding score in Kcal/mol.
    """
    II = []
    JJ = []
    XE = []
    XN = []
    NL = []
    score = []
    cnt = 0
    for index in IND:
        II.extend([lig['atom_connect'][index][:, 0].long() - 1 + cnt])
        JJ.extend([lig['atom_connect'][index][:, 1].long() - 1 + cnt])
        XE.append(lig['bond_type'][index].t().float())

        # feats = []
        # a_feats_idx = torch.nonzero(lig['atom_types'][index])[:, 1]
        # for idx in a_feats_idx:
        #     tmp = torch.tensor(atom_feats[idx.item()])
        #     tmp[0] = tmp[0] * 0.1
        #     tmp[2] = tmp[2] * 0.01
        #     tmp[3] = tmp[3] * 100
        #     feats.append(tmp)
        # feats = torch.vstack(feats)

        XN.append(torch.vstack((lig['atom_types'][index].t().float(),
                                5 * lig['charges'][index].unsqueeze(0))))
        NL.append(XN[-1].shape[1])
        cnt += NL[-1]
        score.append(lig['scores'][index])

    II = torch.hstack(II).to(device)
    JJ = torch.hstack(JJ).to(device)
    XE = torch.hstack(XE).unsqueeze(0).to(device)
    XN = torch.hstack(XN).unsqueeze(0).to(device)
    NL = torch.tensor(NL, device=device)
    score = torch.vstack(score).to(device)
    return II, JJ, XN, XE, NL, score


def loader(num_samples, b_size, shuffle=True):
    """Generate bacthes of random indices to behave as a dataloader.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    b_size : int
        Batch size.
    shuffle : bool, True by deffault
        Permute batches if True, keep order otherwise.
    Returns
    -------
    batches : list
        A list of lists, each of size b_size. The last batch may
        contain exaclty b_size samples or the remaining number of samples.
    """
    if shuffle:
        rng = np.random.default_rng()
        idx = rng.permutation(num_samples).tolist()
    else:
        idx = list(range(num_samples))
    num_batches = num_samples // b_size
    remaining = num_samples % b_size
    batches = [idx[b * b_size: (b + 1)*b_size] for b in range(num_batches)]
    batches.append(idx[-remaining:]) if remaining != 0 else batches
    return batches
