import awkward1 as ak
import numpy as np
import vector
import numba as nb
import dask.dataframe as dd
import dask.array as da
"""
        Takes a DataFrame and converts it into a Awkward array representation
        with features relevant to our model.

        :param df: Pandas DataFrame, The DataFrame with all the momenta-energy coordinates for all the particles
        :param start: int, First element of the DataFrame
        :param stop: int, Last element of the DataFrame
        :return v: OrderedDict, A Ordered Dictionary with all properties of interest



        Here the function is just computing 4 quantities of interest:
        * Eta value relative to the jet
        * Phi value relative to the jet
        * Transverse Momentum of the Particle (log of it)
        * Energy of the Particle (log of it)
"""


# TODO: Compute as many properties as required, keep scope for more always. Maybe use some of it to transform between latent spaces and some for Message Passing.
# TODO: Initially, simply use the low-level features for everything and see how the model trained on low level features compares to model with domain knowledge.

@nb.jit(nopython=True)
def compute_features(v, px, py, pz, energy, label, jet_p4):
    v['label'] = np.stack((label, 1-label), axis=-1)
    v['part_pt_log'] = np.log(jet_p4.pt())
    v['part_e_log'] = np.log(energy)
    v['part_etarel'] = (jet_p4.eta() - jet_p4.eta()) * np.sign(jet_p4.eta())
    v['part_phirel'] = jet_p4.pseudorapidity(jet_p4)


def _transform(df, start=0, stop=-1):

    # Convert the Pandas DataFrame to a Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=10)

    # generate the column list to be extracted
    def _col_list(prefix, max_particles=200):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]

    def process_partition(partition):
        from collections import OrderedDict
        v = OrderedDict()
        _px = partition[_col_list(prefix='PX')].to_dask_array(lengths=True)
        _py = partition[_col_list(prefix='PY')].to_dask_array(lengths=True)
        _pz = partition[_col_list(prefix='PZ')].to_dask_array(lengths=True)
        _e = partition[_col_list(prefix='E')].to_dask_array(lengths=True)

        mask = _e > 0
        n_particles = mask.sum(axis=1)

        jet_px = (_px * mask).sum(axis=1)
        jet_py = (_py * mask).sum(axis=1)
        jet_pz = (_pz * mask).sum(axis=1)
        jet_energy = (_e * mask).sum(axis=1)

        jet_pt = da.sqrt(jet_px ** 2 + jet_py ** 2)
        log_jet_pt = da.log(jet_pt + 1e-6)
        log_jet_energy = da.log(jet_energy + 1e-6)

        jet_eta = 0.5 * da.log((da.sqrt(jet_px ** 2 + jet_py ** 2 + jet_pz ** 2) + jet_pz + 1e-6) /
                               (da.sqrt(jet_px ** 2 + jet_py ** 2 + jet_pz ** 2) - jet_pz + 1e-6))

        px_masked = _px[mask]
        py_masked = _py[mask]
        pz_masked = _pz[mask]
        part_eta = 0.5 * da.log((da.sqrt(px_masked ** 2 + py_masked ** 2 + pz_masked ** 2) + pz_masked + 1e-6) /
                                (da.sqrt(px_masked ** 2 + py_masked ** 2 + pz_masked ** 2) - pz_masked + 1e-6))

        eta_rel = part_eta - jet_eta[:, None]

        # Store results in OrderedDict
        v['jet_pt'] = jet_pt
        v['jet_log_pt'] = log_jet_pt
        v['jet_eta'] = jet_eta
        v['log_jet_energy'] = log_jet_energy
        v['eta_rel'] = eta_rel
        v['n_parts'] = n_particles
        v['label'] = da.stack((partition['is_signal_new'].to_dask_array(lengths=True),
                               1 - partition['is_signal_new'].to_dask_array(lengths=True)), axis=-1)
        v['train_val_test'] = partition['ttv'].to_dask_array(lengths=True)

        del jet_px, jet_py, jet_pz, jet_energy
        del _px, _py, _pz, _e
        del mask, n_particles
        del eta_rel, part_eta
        del px_masked, py_masked, pz_masked
        del jet_pt, log_jet_pt, log_jet_energy
        return v

    # APplying function to each of the partitions
    results = ddf.map_partitions(process_partition, meta = v)

    return results


