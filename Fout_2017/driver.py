"""Fout et al 2017.

Data: https://zenodo.org/record/1127774#.YS6qTHVKhhG
Code: https://github.com/fouticus/pipgcn

Train data:
element 0 is a list of length 175 containing the PDB codes from the
docking benchmark dataset.
element 1 is a list of length 175 containing features for each protein.
Each element is a dictionary containing the following keys:
    - l_vertex: vertex (residue) features for the ligand. analogous to above,
    with shape (y, 70) where y is the number of residues in the ligand.
    - l_edge: edge features for the neighborhood around each residue in the
    ligand. numpy array of shape (y, 20, 2) where y is defined as above. The
    second dimension is the edges to the 20 nearest neighboring residues,
    ordered by decreasing distance. The third dimension allows for two
    features per edge.
    - l_hood_indices: the index of the 20 closest residues to each residue,
    ordered by decreasing distance. numpy array of shape (y, 20, 1). "Index"
    means which row in l_vertex gives the vertex features for the closest
    neighbor, second closest neighbor, etc.
    - r_vertex: vertex (residue) features for the receptor. numpy array of shape
    (x, 70) where x is the number of residues in the receptor and 70 is the
    number of features.
    - r_edge: edge features for the neighborhood around each residue in the
    receptor. numpy array of shape (x, 20, 2) where x is as above.
    - r_hood_indices: analogous to above, shape (x, 20, 1).
    - label: 1 or -1 label for each residue pair. numpy array of shape (x*y, 3).
    Each row looks like (i, j, k) where i is the index of the ligand residue,
    j is the index of the receptor residue, and k is either -1
    (negative example) or 1 (positive example).
    - complex_code: PDB code of the complex. matches the list of codes
    described above.
"""

import torch
import torch.nn as nn
import pickle


class Dense(nn.Module):
    """Fully connected feed forward network."""

    def __init__(self, in_feats, out_feats):
        """Linear and non linear transformations."""
        super(Dense, self).__init__()
        self.stdv = 1e-2
        self.relu = nn.ReLU()
        self.A1 = nn.Parameter(torch.randn(2 * in_feats, in_feats) * self.stdv)
        self.A2 = nn.Parameter(torch.randn(in_feats, 2 * in_feats) * self.stdv)
        self.A3 = nn.Parameter(torch.randn(out_feats, in_feats) * self.stdv)

    def forward(self, x):
        """Apply transformations."""
        xout = self.relu(self.A1 @ x.t())
        xout = self.relu(self.A2 @ xout)
        xout = self.relu(self.A3 @ xout)
        return xout


class Conv_block(nn.Module):
    """Spatial graph convolution operation."""

    def __init__(self, xn_in, xe_in, xn_out):
        """Graph convolution block trainable parameters."""
        super(Conv_block, self).__init__()
        self.stdv = 1e-2
        self.relu = nn.ReLU()
        self.Wc = nn.Parameter(torch.randn(xn_out, xn_in) * self.stdv)  # For center node
        self.Wn = nn.Parameter(torch.randn(xn_out, xn_in) * self.stdv)  # For neighboring nodes
        self.We = nn.Parameter(torch.randn(xn_out, xe_in) * self.stdv)  # For edge features

    def forward(self, x, e, ij):
        """Output of spatial convolution."""
        xnew = torch.empty((x.shape[0], self.Wc.shape[0]), dtype=torch.float)
        for center in range(ij.shape[0]):               # For all nodes in the graph
            nhbrs = ij[center, :, 0]                    # Neighbourhood around center
            xni = x[center, :]                          # Center vertex
            xnj = x[nhbrs].sum(dim=0) / nhbrs.numel()   # Neighborhood (vertex feats)
            xej = e[center].sum(dim=0) / nhbrs.numel()  # Neighborhood (edge feats)
            xout = self.Wc @ xni + self.Wn @ xnj + self.We @ xej
            xnew[center, :] = self.relu(xout)

        return xnew


class GCNN(nn.Module):
    """Gaph Neural Network."""

    def __init__(self, xn, xe, nopen, nhidden, nclose):
        """Make graph net object and initialize trainable parameters."""
        super(GCNN, self).__init__()

        # Note: can use nn.Seq for an arbitrary number of conv blocks
        # Weights are shared between the ligand and receptor legs of the net.
        self.conv1 = Conv_block(xn, xe, nopen)
        self.conv2 = Conv_block(nopen, xe, nopen)
        self.conv3 = Conv_block(nopen, xe, nhidden)
        self.dense = Dense(nhidden, nclose)

    def merge(self, xnr, xnl):
        """Merge ligand and receptor (residue pairwise combinations)."""
        xmerge = torch.empty((xnr.shape[0] * xnl.shape[0], xnr.shape[1]))
        row = 0
        for ii in range(xnr.shape[0]):
            for jj in range(xnl.shape[0]):
                # Note: This could be some other operation
                xmerge[row, :] = 0.5 * (xnr[ii] + xnl[jj])
                row += 1
        return xmerge

    def forward(self, xnr, xer, ijr, xnl, xel, ijl):
        """Forward pass through the network."""
        if not isinstance(xnr, torch.FloatTensor):
            xnr = xnr.float()
        if not isinstance(xer, torch.FloatTensor):
            xer = xer.float()
        if not isinstance(xnl, torch.FloatTensor):
            xnl = xnl.float()
        if not isinstance(xel, torch.FloatTensor):
            xel = xel.float()

        # Ligand conv blocks (Note: could learn/update edge features too)
        xnl = self.conv1(xnl, xel, ijl)
        xnl = self.conv2(xnl, xel, ijl)
        xnl = self.conv3(xnl, xel, ijl)

        # Receptor conv blocks (Note: could learn/update edge features too)
        xnr = self.conv1(xnr, xer, ijr)
        xnr = self.conv2(xnr, xer, ijr)
        xnr = self.conv3(xnr, xer, ijr)

        # Merge rececptor and ligand conv block outputs
        xout = self.dense(self.merge(xnl, xnr))
        return xout


if __name__ == '__main__':

    with open('data/train.cpkl', 'rb') as f:
        data = pickle.load(f, encoding='latin')

    # Receptor
    xnr = torch.from_numpy(data[1][0]["r_vertex"])
    xer = torch.from_numpy(data[1][0]["r_edge"])
    ijr = torch.from_numpy(data[1][0]["r_hood_indices"])
    xn = xnr.shape[1]
    xe = xer.shape[2]

    # Ligand
    xnl = torch.from_numpy(data[1][0]["l_vertex"])
    xel = torch.from_numpy(data[1][0]["l_edge"])
    ijl = torch.from_numpy(data[1][0]["l_hood_indices"])

    # Check output on single sample. To do: test on GPU
    model = GCNN(xn, xe, nopen=32, nhidden=64, nclose=1)
    output = model(xnr, xer, ijr, xnl, xel, ijl)  # [1, |xnl[0]| * |xnr[0]|]

    assert(output.shape == torch.Size([1, xnr.shape[0] * xnl.shape[0]]))
