
from datasets.transformation import transform_gen as T 


def build_transform_gen(config, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    tfm_gens = []
    
    if is_train:
        tfm_gens.append(T.RandomFlip(prob=0.5))
        tfm_gens.append(T.RandomBrightness(0.6,1.4))
        tfm_gens.append(T.RandomSaturation(0.6, 1.4))
        tfm_gens.append(T.RandomLighting(1.))
        
    return tfm_gens
