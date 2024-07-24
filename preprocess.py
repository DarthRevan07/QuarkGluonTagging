import awkward1 as ak
import numpy as np
import vector
import numba as nb

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
    from collections import OrderedDict
    v = OrderedDict()

    def _col_list(prefix, max_particles=200):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]

    df = df.iloc[start:stop]
    # We take the values in the dataframe for all particles of a single event in each row
    # px, py, pz, e are in separate arrays
    print(df[_col_list('PX')])

    _px = df[_col_list(prefix = 'PX')].values
    _py = df[_col_list(prefix = 'PY')].values
    _pz = df[_col_list(prefix = 'PZ')].values
    _e = df[_col_list(prefix = 'E')].values

    # We filter out the non-0 non-negative energy particles
    mask = _e > 0
    n_particles = np.sum(mask, axis=1) # Number of particles for each event where energy is greater than 0
    # _p[mask] filters out the >0 energy particles, and flattens them, so that they can be recollected for each event from counts array.
    px = ak.JaggedArray.fromcounts(n_particles, _px[mask])
    py = ak.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = ak.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = ak.JaggedArray.fromcounts(n_particles, _e[mask])
    # These are jagged arrays with each row for 1 event, and all particles in the row

    # List comprehension mom_objects
    mom_objects = [vector.MomentumObject4D(px = px[i], py = py[i], pz = pz[i], energy = energy[i]) for i in range(len(px))]
    jet_p4 = vector.MomentumObject4D.sum(mom_objects)

    _label = df['is_signal_new'].values
    compute_features(v, px, py, pz, energy, _label, jet_p4)

    del px, py, pz, energy, _px, _py, _pz, _e, mom_objects, df
    return v

