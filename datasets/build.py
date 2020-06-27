
from . import transforms as T


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    
    tfm_gens = []
    
    if is_train:
        tfm_gens.append(T.RandomFlip())
        
    return tfm_gens
