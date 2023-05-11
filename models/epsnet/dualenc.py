import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm

from utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, generate_symmetric_edge_noise, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, get_angle, get_dihedral, eq_transform
from .diffusion import get_timestep_embedding, get_beta_schedule, nonlinearity
import pdb
from copy import deepcopy
from rdkit.Chem import rdMolAlign as MA
from statistics import mean, stdev


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DualEncoderEpsNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        self.hidden_dim = config.hidden_dim
        '''
        timestep embedding
        '''
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
             torch.nn.Linear(config.hidden_dim,
                             config.hidden_dim*4),
             torch.nn.Linear(config.hidden_dim*4,
                             config.hidden_dim*4),
         ])
        self.temb_proj = torch.nn.Linear(config.hidden_dim*4,
                                          config.hidden_dim)
        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.cutoff,
            smooth=config.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.hidden_dim,
            num_convs=config.num_convs_local,
        )

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        '''
        Incorporate parameters together
        '''
        self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])
        self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'

        if self.model_type == 'diffusion':
            # denoising diffusion
            ## betas
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            betas = torch.from_numpy(betas).float()
            self.betas = nn.Parameter(betas, requires_grad=False)
            ## variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            self.num_timesteps = self.betas.size(0)
        elif self.model_type == 'dsm':
            # denoising score matching
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(config.sigma_begin), np.log(config.sigma_end),
                                config.num_noise_level)), dtype=torch.float32)
            self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)
            self.num_timesteps = self.sigmas.size(0)  # betas.shape[0]


    def forward(self, atom_type, pos, R_G, P_G, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, time_step, num_nodes_per_graph, rfp, pfp, dfp,
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True, is_sidechain=None):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
            edge_index_reac, edge_type_reac = extend_graph_order_radius(
                num_nodes=N,
                pos=R_G,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length_reac = get_distance(R_G, edge_index_reac).unsqueeze(-1)   # (E, 1)
            edge_index_prod, edge_type_prod = extend_graph_order_radius(
                num_nodes=N,
                pos=P_G,
                edge_index=bond_index_prod,
                edge_type=bond_type_prod,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length_prod = get_distance(P_G, edge_index_prod).unsqueeze(-1)   # (E, 1)
        local_edge_mask = is_local_edge(edge_type)  # (E, )

        # Emb time_step
        if self.model_type == 'dsm':
            noise_levels = self.sigmas.index_select(0, time_step)  # (G, )
            # # timestep embedding
            # temb = get_timestep_embedding(time_step, self.hidden_dim)
            # temb = self.temb.dense[0](temb)
            # temb = nonlinearity(temb)
            # temb = self.temb.dense[1](temb)
            # temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
            # from graph to node/edge level emb
            node2graph = batch
            edge2graph = node2graph.index_select(0, edge_index[0])
            sigma_edge = noise_levels.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)
            # temb_edge = temb.index_select(0, edge2graph)  # (E , dim)
        elif self.model_type == 'diffusion':
            # with the parameterization of NCSNv2
            # DDPM loss implicit handle the noise variance scale conditioning
            sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)  # (E, 1)


        # Timestep embedding from the GeoDiff of old
        # # timestep embedding
        temb = get_timestep_embedding(time_step, self.hidden_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
        # from graph to node/edge level emb
        node2graph = batch
        edge2graph = node2graph.index_select(0, edge_index[0])
        temb_edge = temb.index_select(0, edge2graph)  # (E , dim)

        # Routine to calculate how many edges below to each molecule
        num_edges_per_graph = num_nodes_per_graph*(num_nodes_per_graph-1)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type,
            time_step=time_step,
            edge2graph=edge2graph,
            rfp=rfp,
            pfp=pfp,
            dfp=dfp,
            num_nodes_per_graph=num_nodes_per_graph
        )   # Embed edges
        edge_attr_global += temb_edge

        # Encoding global reactant
        edge_attr_global_reac = self.edge_encoder_global(
            edge_length=edge_length_reac,
            edge_type=edge_type_reac,
            time_step=time_step,
            edge2graph=edge2graph,
            rfp=rfp,
            pfp=pfp,
            dfp=dfp,
            num_nodes_per_graph=num_nodes_per_graph
        )   # Embed edges
        edge_attr_global_reac += temb_edge

        # Encoding global product
        edge_attr_global_prod = self.edge_encoder_global(
            edge_length=edge_length_prod,
            edge_type=edge_type_prod,
            time_step=time_step,
            edge2graph=edge2graph,
            rfp=rfp,
            pfp=pfp,
            dfp=dfp,
            num_nodes_per_graph=num_nodes_per_graph
        )   # Embed edges
        edge_attr_global_prod += temb_edge

        # Global
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        # Global reactant
        node_attr_global_reac = self.encoder_global(
            z=atom_type,
            edge_index=edge_index_reac,
            edge_length=edge_length_reac,
            edge_attr=edge_attr_global_reac,
        )
        # Global product
        node_attr_global_prod = self.encoder_global(
            z=atom_type,
            edge_index=edge_index_prod,
            edge_length=edge_length_prod,
            edge_attr=edge_attr_global_prod,
        )
        ## Assemble pairwise features
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )    # (E_global, 2H)
        ## Assemble pairwise features reactant
        h_pair_global_reac = assemble_atom_pair_feature(
            node_attr=node_attr_global_reac,
            edge_index=edge_index_reac,
            edge_attr=edge_attr_global_reac,
        )    # (E_global, 2H)
        ## Assemble pairwise features product
        h_pair_global_prod = assemble_atom_pair_feature(
            node_attr=node_attr_global_prod,
            edge_index=edge_index_prod,
            edge_attr=edge_attr_global_prod,
        )    # (E_global, 2H)

        # Combine encoding of TS, reactant, and product
        h_pair_global = h_pair_global + h_pair_global_reac + h_pair_global_prod

        ## Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)    # (E_global, 1)
        
        # Encoding local
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type,
            time_step=time_step,
            edge2graph=edge2graph,
            rfp=rfp,
            pfp=pfp,
            dfp=dfp,
            num_nodes_per_graph=num_nodes_per_graph
        )   # Embed edges
        # edge_attr += temb_edge

        # Local
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        ## Assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )    # (E_local, 2H)
        ## Invariant features of edges (bond graph, local)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask]) # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge) # (E_local, 1)

        if return_edges:
            return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        else:
            return edge_inv_global, edge_inv_local
        
    def find_rigid_alignment(self, A, B):
        """
        See: https://en.wikipedia.org/wiki/Kabsch_algorithm
        2-D or 3-D registration with known correspondences.
        Registration occurs in the zero centered coordinate system, and then
        must be transported back.
            Args:
            -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
            -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
            Returns:
            -    R: optimal rotation
            -    t: optimal translation
        Test on rotation + translation and on rotation + translation + reflection
            >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
            >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
            >>> B = (R0.mm(A.T)).T
            >>> t0 = torch.tensor([3., 3.])
            >>> B += t0
            >>> R, t = find_rigid_alignment(A, B)
            >>> A_aligned = (R.mm(A.T)).T + t
            >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
            >>> rmsd
            tensor(3.7064e-07)
            >>> B *= torch.tensor([-1., 1.])
            >>> R, t = find_rigid_alignment(A, B)
            >>> A_aligned = (R.mm(A.T)).T + t
            >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
            >>> rmsd
            tensor(3.7064e-07)
        """
        a_mean = A.mean(axis=0)
        b_mean = B.mean(axis=0)
        A_c = A - a_mean
        B_c = B - b_mean
        # Covariance matrix
        H = A_c.T.mm(B_c)
        U, S, V = torch.svd(H)
        # Rotation matrix
        R = V.mm(U.T)
        # Translation vector
        t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
        t = t.T
        return R, t.squeeze()
    

    def get_loss(self, mol, atom_type, pos, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, num_nodes_per_graph, num_graphs, R_G, P_G, rfp, pfp, dfp, noise,
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(mol, atom_type, pos, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, num_nodes_per_graph, num_graphs, R_G, P_G, rfp, pfp, dfp, noise,
                anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius, is_sidechain)
        elif self.model_type == 'dsm':
            return self.get_loss_dsm(atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius, is_sidechain)


    def get_loss_diffusion(self, mol, atom_type, pos, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, num_nodes_per_graph, num_graphs, R_G, P_G, rfp, pfp, dfp, noise,
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        N = atom_type.size(0)
        node2graph = batch

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device)
        time_step = torch.cat(
            [time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs]
        a = time_step / self.num_timesteps  # (G, )
        # Perterb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)
        if noise == "gaussian":
            pos_noise = torch.zeros(size=pos.size(), device=pos.device)
            pos_noise.normal_()
        elif noise == "interp":
            pos_noise = (R_G + P_G) / 2
        # Kabsch align the TS to the linear intepolation
        R, t = self.find_rigid_alignment(pos, pos_noise)
        pos_align = (R.mm(pos.T)).T + t
        pos_perturbed = (1-a_pos)*pos_align + a_pos*pos_noise

        # Update invariant edge features, as shown in equation 5-7
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
            atom_type = atom_type,
            pos = pos_perturbed,
            R_G = R_G,
            P_G = P_G,
            bond_index = bond_index,
            bond_type = bond_type,
            bond_index_prod = bond_index_prod,
            bond_type_prod = bond_type_prod,
            batch = batch,
            time_step = time_step,
            num_nodes_per_graph = num_nodes_per_graph,
            rfp = rfp,
            pfp = pfp,
            dfp = dfp,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius,
            is_sidechain = is_sidechain
        )   # (E_global, 1), (E_local, 1)

        # Grab the generated bond distances to generate the global mask
        #d_perturbed = edge_length
        #global_mask = torch.logical_and(
        #                    torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
        #                    ~local_edge_mask.unsqueeze(-1)
        #                )

        # Apply the mask and equivariant transform to the generated edge_inv_global with the perturbed geometry to generate mods to pertubed geometry
        #edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        denoised_pos = pos_perturbed + node_eq_global

        # Rotate and translate the geometry to compare to the transition state
        R, t = self.find_rigid_alignment(denoised_pos, pos)
        score_pos = (R.mm(denoised_pos.T)).T + t

        # Calculate global loss with target geometry and generated geometry
        loss_global = (score_pos - pos)**2
        loss_global = torch.sum(loss_global, dim=-1, keepdim=True)

        benchmark = (pos_perturbed - pos_align)**2
        benchmark_loss = torch.sum(benchmark, dim=-1, keepdim=True)
        benchmark_loss_mean = benchmark_loss.mean()
        ratio = loss_global.mean().item()/benchmark_loss_mean.item()

        return loss_global, loss_global, loss_global, ratio
    
    def set_rdmol_positions(self, rdkit_mol, pos):
        """
        Args:
            rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
            pos: (N_atoms, 3)
        """
        mol = deepcopy(rdkit_mol)
        self.set_rdmol_positions_(mol, pos)
        return mol

    def set_rdmol_positions_(self, mol, pos):
        """
        Args:
            rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
            pos: (N_atoms, 3)
        """
        for i in range(pos.shape[0]):
            mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
        return mol


    def langevin_dynamics_sample(self, truth, mol, atom_type, pos_init, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, num_nodes_per_graph, num_graphs, R_G, P_G, rfp, pfp, dfp, noise, extend_order=True, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        if self.model_type == 'diffusion':
            return self.langevin_dynamics_sample_diffusion(truth, mol, atom_type, pos_init, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, num_nodes_per_graph, num_graphs, R_G, P_G, rfp, pfp, dfp, noise, extend_order, extend_radius, 
                        n_steps, step_lr, clip, clip_local, clip_pos, min_sigma, is_sidechain,
                        global_start_sigma, w_global, w_reg, 
                        sampling_type=kwargs.get("sampling_type", 'ddpm_noisy'), eta=kwargs.get("eta", 1.))
        elif self.model_type == 'dsm':
            return self.langevin_dynamics_sample_dsm(atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius, 
                        n_steps, step_lr, clip, clip_local, clip_pos, min_sigma, is_sidechain,
                        global_start_sigma, w_global, w_reg)

    def langevin_dynamics_sample_diffusion(self, truth, mol, atom_type, pos_init_ugh, bond_index, bond_type, bond_index_prod, bond_type_prod, batch, num_nodes_per_graph, num_graphs, R_G, P_G, rfp, pfp, dfp, noise, extend_order=True, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        pos_traj = []
        with torch.no_grad():
            seq = range(self.num_timesteps-n_steps+1, self.num_timesteps+1)
            seq_next = [0] + list(seq[:-1])
            pos = pos_init_ugh
            #print("RMSD between true geometry and interpolation")
            truth_mol = self.set_rdmol_positions(mol[0], truth.cpu())
            interp_mol = self.set_rdmol_positions(mol[0], pos_init_ugh.cpu())
            dis = MA.GetBestRMS(interp_mol, truth_mol)
            #print(dis)
            RMSD_traj = []
            RMSD_traj.append(dis)
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)
                # Send position through GFN and recover generated geometry
                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                                atom_type=atom_type,
                                pos=pos,
                                R_G = R_G,
                                P_G = P_G,
                                bond_index=bond_index,
                                bond_type=bond_type,
                                bond_index_prod = bond_index_prod,
                                bond_type_prod = bond_type_prod,
                                batch=batch,
                                time_step=t,
                                num_nodes_per_graph=num_nodes_per_graph,
                                rfp = rfp,
                                pfp = pfp,
                                dfp = dfp,
                                return_edges=True,
                                extend_order=extend_order,
                                extend_radius=extend_radius,
                                is_sidechain=is_sidechain
                            )   # (E_global, 1), (E_local, 1)
                gen_pos_mods = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                gen_pos = pos + gen_pos_mods
                
                # Check against the truth for my sanity
                #print("RMSD between true geometry and timestep " + str(i) + " generated TS")
                gen_mol = self.set_rdmol_positions(mol[0], gen_pos)
                other_dis = MA.GetBestRMS(gen_mol, truth_mol)
                #print(other_dis)

                # Calculate amount of noise to add to generated geometry
                next_timestep = torch.full(size=(num_graphs,), fill_value=j, dtype=torch.long, device=pos.device)
                a = next_timestep / self.num_timesteps
                a_pos = a.index_select(0, batch).unsqueeze(-1)

                # Add noise to generated geometry
                if noise == "gaussian":
                    pos_init_ugh = torch.randn(num_nodes_per_graph[0], 3).to(pos.device)

                pos_noised = (1-a_pos)*gen_pos + a_pos*pos_init_ugh
                #print("RMSD between true geometry and noised timestep " + str(j))
                noise_mol = self.set_rdmol_positions(mol[0], pos_noised)
                another_dis = MA.GetBestRMS(noise_mol, truth_mol)
                #print(another_dis)
                RMSD_traj.append(another_dis)
                if another_dis > 10:
                    break

                # I currently have basic sampling implemented here, to implement Algorithm 2, set pos = pos_next
                a_prior = t / self.num_timesteps
                a_prior_pos = a_prior.index_select(0, batch).unsqueeze(-1)
                prior_noised = (1-a_prior_pos)*gen_pos + a_prior_pos*pos_init_ugh
                pos_next = pos - prior_noised + pos_noised
                pos = pos_noised
        return pos, pos_traj, another_dis

    def langevin_dynamics_sample_diffusion_old(self, atom_type, pos_init, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, extend_order, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        if is_sidechain is not None:
            assert pos_gt is not None, 'need crd of backbone for sidechain prediction'
        with torch.no_grad():
            # skip = self.num_timesteps // n_steps
            # seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            
            pos = pos_init * sigmas[-1]
            if is_sidechain is not None:
                pos[~is_sidechain] = pos_gt[~is_sidechain]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    num_nodes_per_graph=num_nodes_per_graph,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain
                )   # (E_global, 1), (E_local, 1)

                # Local
                node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                # Global
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1-local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                # Sum
                eps_pos = node_eq_local + node_eq_global * w_global # + eps_pos_reg * w_reg

                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(pos)  #  center_pos(torch.randn_like(pos), batch)
                if sampling_type == 'generalized' or sampling_type == 'ddpm_noisy':
                    b = self.betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    if sampling_type == 'generalized':
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos
                        ## original
                        # pos0_t = (pos - et * (1 - at).sqrt()) / at.sqrt()
                        ## reweighted
                        # pos0_t = pos - et * (1 - at).sqrt() / at.sqrt()
                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        # pos_next = at_next.sqrt() * pos0_t + c1 * noise + c2 * et
                        # pos_next = pos0_t + c1 * noise / at_next.sqrt() + c2 * et / at_next.sqrt()

                        # pos_next = pos + et * (c2 / at_next.sqrt() - (1 - at).sqrt() / at.sqrt()) + noise * c1 / at_next.sqrt()
                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 5 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld<step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld<step_size_noise_generalized else step_size_noise_generalized

                        pos_next = pos - et * step_size_pos +  noise * step_size_noise

                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size*2)

                pos = pos_next

                if is_sidechain is not None:
                    pos[~is_sidechain] = pos_gt[~is_sidechain]

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
            
        return pos, pos_traj
    

    def get_loss_dsm(self, atom_type, pos, bond_index, bond_type, batch, num_nodes_per_graph, num_graphs, 
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        N = atom_type.size(0)
        node2graph = batch

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels (sigmas)
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device)
        time_step = torch.cat(
            [time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs]
        noise_levels = self.sigmas.index_select(0, time_step)  # (G, )
        # Perterb pos
        sigmas_pos = noise_levels.index_select(0, node2graph).unsqueeze(-1)  # (E, 1)
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * sigmas_pos

        # Update invariant edge features, as shown in equation 5-7
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
            atom_type = atom_type,
            pos = pos_perturbed,
            bond_index = bond_index,
            bond_type = bond_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius,
            is_sidechain = is_sidechain
        )   # (E_global, 1), (E_local, 1)

        edge2graph = node2graph.index_select(0, edge_index[0])
        # Compute sigmas_edge
        sigmas_edge = noise_levels.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        d_perturbed = edge_length
        # Filtering for protein
        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0 
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))
        d_target = 1. / (sigmas_edge ** 2) * (d_gt - d_perturbed)   # (E_global, 1), denoising direction

        global_mask = torch.logical_and(
                            torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
                            ~local_edge_mask.unsqueeze(-1)
                        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = 0.5 * ((node_eq_global - target_pos_global)**2) * (sigmas_pos ** anneal_power)
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)

        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = 0.5 * ((node_eq_local - target_pos_local)**2) * (sigmas_pos ** anneal_power)
        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True)

        # loss for atomic eps regression
        loss = loss_global + loss_local
        # loss_pos = scatter_add(loss_pos.squeeze(), node2graph)  # (G, 1)

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss


    def langevin_dynamics_sample_dsm(self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True, 
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0):


        sigmas = self.sigmas
        pos_traj = []
        if is_sidechain is not None:
            assert pos_gt is not None, 'need crd of backbone for sidechain prediction'
        with torch.no_grad():
            pos = pos_init
            if is_sidechain is not None:
                pos[~is_sidechain] = pos_gt[~is_sidechain]
            for i, sigma in enumerate(tqdm(sigmas, desc='sample')):
                if sigma < min_sigma:
                    break
                time_step = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for step in range(n_steps):
                    edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                        atom_type=atom_type,
                        pos=pos,
                        bond_index=bond_index,
                        bond_type=bond_type,
                        batch=batch,
                        time_step=time_step,
                        return_edges=True,
                        extend_order=extend_order,
                        extend_radius=extend_radius,
                        is_sidechain=is_sidechain
                    )   # (E_global, 1), (E_local, 1)

                    # Local
                    node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                    # Global
                    if sigma < global_start_sigma:
                        edge_inv_global = edge_inv_global * (1-local_edge_mask.view(-1, 1).float())
                        node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                        node_eq_global = clip_norm(node_eq_global, limit=clip)
                    else:
                        node_eq_global = 0
                    # Sum
                    eps_pos = node_eq_local + node_eq_global * w_global # + eps_pos_reg * w_reg

                    # Update
                    noise = torch.randn_like(pos) * torch.sqrt(step_size*2)
                    pos = pos + step_size * eps_pos + noise
                    if is_sidechain is not None:
                        pos[~is_sidechain] = pos_gt[~is_sidechain]

                    if torch.isnan(pos).any():
                        print('NaN detected. Please restart.')
                        raise FloatingPointError()
                    pos = center_pos(pos, batch)
                    if clip_pos is not None:
                        pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                    pos_traj.append(pos.clone().cpu())
            
        return pos, pos_traj



def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
