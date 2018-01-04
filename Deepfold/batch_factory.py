# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import print_function
import numpy as np
import random
from . import grid
import os

def get_batch(start_index, end_index, *values):
    '''Slices in multiple arrays at once'''
    values_batch = []
    for value in values:
        if value is None:
            values_batch.append(None)
        else:
            values_batch.append(value[start_index:end_index])
    return values_batch

class ProteinData:
    '''Training data for a single protein, summarizing info at protein and residue level'''

    def __init__(self, protein_feature_filename, key_filter=[]):
        '''Constructor'''
        
        self.features = {}

        # Retrieve data from npz file
        protein_loader = np.load(protein_feature_filename)

        # Extract relevant keys
        self.seq_length = 0
        selected_feature_keys = []
        self.dimensions = 0
        for key in protein_loader.keys():
            value = protein_loader[key]
            # Scalars are dealt with separately
            if len(value.shape) == 0:
                value = np.asscalar(value)
            else:
                if key in key_filter:
                    selected_feature_keys.append(key)
            self.features[key] = value

        # One hot encoding of amino acid sequence used to set length
        if "aa_one_hot" in protein_loader.keys():
            self.seq_length = protein_loader["aa_one_hot"].shape[0]

        # Unfold boundary indices value so that each residue knows its chain id
        if "chain_boundary_indices" in protein_loader.keys():
            chain_boundaries = self.features["chain_boundary_indices"]
            chain_ids = self.features["chain_ids"]
            chain_values = []
            for chain_index, boundary in enumerate(chain_boundaries[1:]):
                length = boundary
                if chain_index > 0:
                    length -= chain_boundaries[1:][chain_index-1]
                chain_values += [chain_ids[chain_index]]*length
            self.features["chain_ids"] = np.array(chain_values, dtype='a5')

        
        if len(selected_feature_keys) > 0:
            self.selected_features = self.features[selected_feature_keys[0]]
        for i in range(1, len(selected_feature_keys)):
            self.selected_features = np.vstack([self.selected_features,
                                                self.features[selected_feature_keys[i]]])

    def initialize_residue_features(self, size):
        '''Initialize grid array'''
        return np.zeros([size]+list(self.selected_features.shape[1:]))
        
    def __len__(self):
        return self.seq_length
        
    def fetch_residue_features(self):
        pass

    def forget_residue_features(self):
        pass

    def get_residue_features(self, residue_index):
        return self.selected_features[residue_index]
        
    
class ProteinGridData(ProteinData):
    '''Training data for a single protein, specialized for data in a ND grid layout'''
    
    def __init__(self, protein_feature_filename, grid_feature_filename, max_sequence_distance=15, duplicate_origin=False):
        '''Constructor'''
        
        ProteinData.__init__(self, protein_feature_filename)

        self.grid_feature_filename = grid_feature_filename

        self.coordinate_system = grid.CoordinateSystem(self.features.get("coordinate_system", grid.CoordinateSystem.spherical.value))

        self.grid_shape = grid.get_grid_shape_map[self.coordinate_system](max_radius=self.features["max_radius"],
                                                                          n_features=self.features["n_features"],
                                                                          bins_per_angstrom=self.features["bins_per_angstrom"])

        if isinstance(max_sequence_distance, list):
            assert(False)
        self.selector_array = None
        self.indices_array = None
        self.max_sequence_distance = max_sequence_distance

        self.duplicate_origin = duplicate_origin

    def initialize_residue_features(self, size):
        '''Initialize grid array'''
        return np.zeros((size,) + self.grid_shape)
        
    def fetch_residue_features(self):
        '''Read in residue information. This takes up quite some space, and is therefore
           not done by default during construction'''
        if self.selector_array is None:
            residue_loader = np.load(self.grid_feature_filename)
            self.selector_array = residue_loader["selector"]
            self.indices_array = residue_loader["indices"]

    def forget_residue_features(self):
        '''Forget about residue information to free up space in memort.'''
        del self.selector_array
        del self.indices_array
        self.selector_array = None
        self.indices_array = None
        
    def get_residue_features(self, residue_index):
        '''Construct grid matrix from residue features'''

        # If selector_array has not been set, we fetch it here, but remember
        # to remove it again at the end of the function.
        fetch_temporarily = False
        if self.selector_array is None:
            self.fetch_grid_features()
            fetch_temporarily = True

        # Extract information for the current residue index
        selector = self.selector_array[residue_index][self.selector_array[residue_index]>=0]

        # Extract data on which grid indices to set
        indices = self.indices_array[residue_index][self.indices_array[residue_index][:,0]>=0]

        # Create grid
        grid_matrix = grid.create_grid_map[self.coordinate_system](max_radius=self.features["max_radius"],
                                                                   n_features=self.features["n_features"],
                                                                   bins_per_angstrom=self.features["bins_per_angstrom"])

        start_index = 0
        for feature_name in self.features["residue_features"]:

            if feature_name == "residue_index":
                chain_index = np.searchsorted(self.features["chain_boundary_indices"], residue_index, side='right')-1

                chain_selector = np.logical_and(self.features[feature_name] >= self.features["chain_boundary_indices"][chain_index],
                                                self.features[feature_name] < self.features["chain_boundary_indices"][chain_index+1])
                
                full_feature = self.features[feature_name] - residue_index
                full_feature = np.clip(full_feature, -self.max_sequence_distance, self.max_sequence_distance)
                full_feature[np.logical_not(chain_selector)] = self.max_sequence_distance

            else :
                full_feature = self.features[feature_name]
            feature = full_feature[selector]

            end_index = start_index + feature.shape[1]

            # Get default values if they are available
            if (feature_name+"_default") in self.features:
                grid_matrix[[slice(None)]*(grid_matrix.ndim-1) + [slice(start_index, end_index)]] = self.features[feature_name + "_default"]

            if feature_name == "residue_index":
                grid_matrix[[slice(None)]*(grid_matrix.ndim-1) + [slice(start_index, end_index)]] = self.max_sequence_distance+1

            grid_matrix[list(indices.T) + [slice(start_index,end_index)]] = feature

            start_index += feature.shape[1]

        # Forget residue features again, if necessary
        if fetch_temporarily:
            self.forget_grid_features()
        
        return grid_matrix
    
        
class BatchFactory:
    '''Create batches for training
    
    You can add several datasets to a batchfactory by key. 
    When calling next() the returned batch will be a dictionary with key,value pairs 
    of these keys and a batch within the corresponding data set
    '''
    
    def __init__(self):
        '''Constructor'''

        # pdb_id -> ProteinData map 
        self.features = {}

        # List of (pdb_ids, res_index) pairs 
        self.features_expanded = []

        # Current index into features_expanded list
        self.feature_index = 0

        # Keep track of completed cycles through data
        self.epoch_count = 0

    def add_data_set(self, key, protein_feature_filenames, grid_feature_filenames=None, key_filter=[], duplicate_origin=False):
        '''Add a new dataset from a list of filenames. These dataset will be represented in each batch - accessible by their key value'''
        
        if grid_feature_filenames is None:
            grid_feature_filenames = [None]*len(protein_feature_filenames)

        for protein_feature_filename, grid_feature_filename in zip(sorted(protein_feature_filenames, key=lambda x: x if x is not None else "ZZZ"),
                                                                   sorted(grid_feature_filenames, key=lambda x: x if x is not None else "ZZZ")):

            pdb_id = os.path.basename(protein_feature_filename)[0:5]

            # Check whether different datasets refer to the same PDB entry
            if grid_feature_filename is not None:
                # Test that protein and residue data files have the same pdb_id prefix
                if (pdb_id != os.path.basename(grid_feature_filename)[0:5]):
                    raise ValueError("%s != %s: Mismatch in protein and residue feature filenames (one of them is probably missing)" % (pdb_id, os.path.basename(grid_feature_filename)[0:5]))

            # Create feature data
            if grid_feature_filename is not None:
                protein_data = ProteinGridData(protein_feature_filename, grid_feature_filename, duplicate_origin)
            else:
                protein_data = ProteinData(protein_feature_filename, key_filter)
            if pdb_id not in self.features:
                self.features[pdb_id] = {}
            self.features[pdb_id][key] = protein_data

        # Randomize order of proteins
        self.shuffle_features()

    def data_size(self):
        return len(self.features_expanded)
            
    def shuffle_features(self):
        '''Randomize order of pdb_ids'''

        # Randomize order
        feature_pdb_ids = list(self.features.keys())
        random.shuffle(feature_pdb_ids)

        # Repopulate self.features_expanded
        self.features_expanded = []
        for pdb_id in feature_pdb_ids:
            n_residues = len(list(self.features[pdb_id].values())[0])
            self.features_expanded += zip([pdb_id]*n_residues, range(n_residues))

        # Reset index counter
        self.feature_index = 0

    def next(self, max_size=10, enforce_protein_boundaries=True, subbatch_max_size=None, increment_counter=True, include_pdb_ids=False, return_single_proteins=False):
        '''Create next batch.
           Optionally, the batch can be divided into subbatches.
        '''

        subbatch_sizes = None
        size = max_size
        if subbatch_max_size is not None:
            subbatch_sizes = []
            if enforce_protein_boundaries:
                pdb_ids = []
                for i in range(max_size):
                    index = (self.feature_index+i) % len(self.features_expanded)
                    pdb_ids.append(self.features_expanded[index][0])
                indices = sorted(np.unique(pdb_ids, return_index=True)[1])

                # if the pdb_index changes right at the boundary, or if the entry spans the entire range, we can include the last entry
                last_entry_index = (self.feature_index+max_size-1) % len(self.features_expanded)
                last_entry_index_next = (self.feature_index+max_size) % len(self.features_expanded)
                if (len(indices) == 1 or
                    self.features_expanded[last_entry_index][0] !=
                    self.features_expanded[last_entry_index_next][0]):
                    # include last entry
                    indices.append(max_size)

                size = indices[-1]
                max_index = len(indices)-1
                if return_single_proteins:
                    size = indices[1]
                    max_index = 1
                for i in range(max_index):
                    index = indices[i]+self.feature_index
                    length = indices[i+1]-indices[i]
                    n_subbatches = (length//subbatch_max_size)
                    if length%subbatch_max_size > 0:
                        n_subbatches += 1
                    subbatch_size_array = np.array([length/n_subbatches]*n_subbatches)
                    remainder = length%n_subbatches
                    subbatch_size_array[np.arange(remainder)] += 1
                    subbatch_sizes += list(subbatch_size_array)

                subbatch_sizes = np.array(subbatch_sizes)
            else:
                n_subbatches = max_size//subbatch_max_size
                if max_size%subbatch_max_size > 0:
                    n_subbatches += 1
                subbatch_size_array = np.array([max_size//n_subbatches]*n_subbatches)
                remainder = max_size%n_subbatches
                subbatch_size_array[np.arange(remainder)] += 1
                subbatch_sizes = subbatch_size_array
            assert(np.sum(subbatch_sizes) == size)

        residue_features = None
        pdb_ids = []
        for i in range(size):

            # Extract ProteinData object
            index = (self.feature_index+i) % len(self.features_expanded)
            pdb_id, residue_index = self.features_expanded[index]

            # Create container if necessary
            # This is done here because the shape is not known until we see the first instance
            if residue_features is None:
                residue_features = {}
                for key in self.features[pdb_id]:
                    residue_features[key] = self.features[pdb_id][key].initialize_residue_features(size)
                
                if include_pdb_ids:
                    residue_features["pdb"] = [None for _ in range(size)]
                
            for key in self.features[pdb_id]:

                # Pre-fetch residue features
                self.features[pdb_id][key].fetch_residue_features()

                # Get residue features
                residue_features_value = self.features[pdb_id][key].get_residue_features(residue_index)
                if residue_features[key][i].dtype is not residue_features_value.dtype:
                    residue_features[key] = residue_features[key].astype(residue_features_value.dtype)
                residue_features[key][i] = residue_features_value

            if include_pdb_ids:
                residue_features["pdb"][i] = pdb_id 
                
            # Keep track of which pdb_ids we have prefetched residue features for
            pdb_ids.append(pdb_id)


        # For all pre-fetched pdb_ids, make ProteinData object forget its residue data
        # (too large to keep it all in memory)
        for pdb_id in pdb_ids:
            for key in self.features[pdb_id]:
                self.features[pdb_id][key].forget_residue_features()

        # Increment counter
        if increment_counter:
            self.feature_index += size

        # If counter passes total number of features, reshuffle and reset
        if self.feature_index >= len(self.features_expanded):
            self.epoch_count += 1
            self.shuffle_features()

        # Make sure that feature_index is correctly set to zero if running on the complete se.
        if max_size == len(self.features_expanded) and not return_single_proteins:
            assert(self.feature_index == 0)
            
        return residue_features, subbatch_sizes


if __name__ == '__main__':

    from argparse import ArgumentParser
    import glob
    import os

    # Command line arguments
    parser = ArgumentParser()
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Location of input files containing features")
    parser.add_argument("--test-set-fraction", dest="test_set_fraction",
                        help="Fraction of data set aside for testing", type=float, default=0.25)
    parser.add_argument("--validation-set-size", dest="validation_set_size",
                        help="Size of validation set (taken out of training set)", type=int, default=10)
    parser.add_argument("--max-batch-size", dest="max_batch_size",
                        help="Maximum batch size used during training", type=int, default=100)
    
    options = parser.parse_args()

    # Read in names of all feature files
    protein_feature_filenames = sorted(glob.glob(os.path.join(options.input_dir, "*protein_features.npz")))
    grid_feature_filenames = sorted(glob.glob(os.path.join(options.input_dir, "*residue_features.npz")))

    # Set range for validation and test set
    validation_end = test_start = int(len(protein_feature_filenames)*(1.-options.test_set_fraction))
    train_end = validation_start = int(validation_end-options.validation_set_size)    

    # Create batch factory and add data sets
    batch_factory = BatchFactory()
    batch_factory.add_data_set("data",
                               protein_feature_filenames[:train_end],
                               grid_feature_filenames[:train_end])

    batch_factory.add_data_set("model_output",
                               protein_feature_filenames[:train_end],
                               key_filter=["aa_one_hot"])

    # Read out total data size
    total_data_size = batch_factory.data_size()

    num_passes = 2
    for i in range(num_passes):

        data_size = 0
        while data_size < total_data_size:

            # Extract the next batch 
            batch, _ = batch_factory.next(options.max_batch_size)

            # From the batch, data and labels can be extracted
            grid_matrix = batch["data"]
            labels = batch["model_output"]


            print(grid_matrix.shape)
            print(labels.shape)
    
