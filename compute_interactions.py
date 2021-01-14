#%%
import tensorflow as tf 
import numpy as np
import sys,os,glob
ahlf_dir = "./AHLF"
sys.path.append(ahlf_dir)


from load_model import model
from get_spectrum_as_numpy import get_spectrum

spectrum = get_spectrum(os.path.join(ahlf_dir,'example/example.mgf'))

from path_explain import PathExplainerTF

@tf.function
def f(x):
    return model(x)

explainer = PathExplainerTF(f)

take_specific_spectrum=0
spectrum=spectrum[take_specific_spectrum,:,:]
spectrum=np.expand_dims(spectrum,0)
background_spectra = np.zeros_like(spectrum)

print(spectrum.shape)

print('calculating interactions:')
two_vec_axis=0
peaks_indices = np.argwhere(spectrum[:,:,two_vec_axis]).flatten()
print(len(peaks_indices))
if len(peaks_indices) < 500:
    for ind in peaks_indices:
        single_interactions = explainer.interactions(inputs=spectrum,
                                            baseline=background_spectra,
                                            batch_size=1,
                                            num_samples=5,
                                            use_expectation=False,
                                            output_indices=0,
                                            interaction_index=(ind,two_vec_axis))
        single_interactions = single_interactions[:,peaks_indices,two_vec_axis]                                   
        single_interactions = np.expand_dims(single_interactions,axis=2)
        try:        
            interactions=np.concatenate([interactions,single_interactions],axis=2)
        except:
            interactions=single_interactions


print(interactions.shape)
print(spectrum.shape)


import matplotlib.pyplot as plt
import seaborn as sns

interactions = np.squeeze(interactions)
np.fill_diagonal(interactions,0)

print(interactions.shape)

sns.heatmap(interactions)
plt.legend()
plt.savefig('interactions.png')
#%%





