import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray


def PreprocessAtariStates(state):
    # state = Image.fromarray(state)
    # resized_state = state.convert('LA').resize((110, 84))
    resized_state = np.expand_dims(resize(rgb2gray(state), (110, 84), anti_aliasing = True), axis = 0)

    resized_state = resized_state.reshape(resized_state.shape[1], resized_state.shape[2])
    croped_state = np.expand_dims(resized_state[15:99, :], axis = 0)

    return croped_state
