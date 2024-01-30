import os
# import voxelmorph with pytorch backend from source
os.environ['NEURITE_BACKEND'] = 'pytorch'  ###############333
os.environ['VXM_BACKEND'] = 'pytorch'      ############
import sys 
sys.path.append('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai')
import voxelmorph as vxm

'''
Script to define the registration network we are going to use
we have the model, the shape at each layer, the saving directory
'''

def vxm_model(model_n,
        device,
        load_model,
        enc,
        dec, 
        in_shape,
        bidir,
        int_steps,
        int_downsize,**kwargs):
    # saving folder
    model_dir = 'models/'+model_n
    os.makedirs(model_dir, exist_ok=True)

    # unet architecture
    enc_nf = enc if enc else [16, 32, 32, 32]
    dec_nf = dec if dec else [32, 32, 32, 32, 32, 16, 16]

    if load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(load_model, device)
    else:
        # otherwise configure new model
        
        model = vxm.networks.VxmDense(
            inshape=in_shape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=int_steps,
            int_downsize=int_downsize, 
            **kwargs)

    return model.to(device), model_dir
