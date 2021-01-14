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

attributions = explainer.attributions(inputs=spectrum,
                                    baseline=background_spectra,
                                    batch_size=1,
                                    num_samples=10,
                                    use_expectation=False)

attributions = np.squeeze(attributions)
spectrum = np.squeeze(spectrum)

print(attributions.shape)
print(spectrum.shape)


import matplotlib.pyplot as plt
plt.stem(spectrum[:,0]/np.max(spectrum[:,0]),linefmt='C0-',markerfmt=' ',use_line_collection=True,label='spectrum')
plt.stem(attributions[:,0]/np.max(np.abs(attributions[:,0])),linefmt='C1-',markerfmt=' ',use_line_collection=True,label='attributions')
plt.legend()
plt.savefig('attributions.png')
#%%





