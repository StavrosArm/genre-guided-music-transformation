import numpy as np
from scipy.linalg import sqrtm


def calculate_fad_distance(
    embeddings_real_rock: np.ndarray, embeddings_distorted_rock: np.ndarray
) -> float:
    """
    Compute the Fréchet Audio Distance (FAD) between two sets of embeddings.

    :param embeddings_real_rock: Embeddings created by real rock music
    :param embeddings_distorted_rock: Embeddings created by distorted rock music
    """

    assert (
        embeddings_real_rock.ndim == 2 and embeddings_distorted_rock.ndim == 2
    ), "Embeddings must be 2D arrays."
    assert (
        embeddings_real_rock.shape[1] == embeddings_distorted_rock.shape[1]
    ), "Embedding dimensions must match."

    mu_real = np.mean(embeddings_real_rock, axis=0)
    sigma_real = np.cov(embeddings_real_rock, rowvar=False)

    mu_distorted = np.mean(embeddings_distorted_rock, axis=0)
    sigma_distorted = np.cov(embeddings_distorted_rock, rowvar=False)

    diff = mu_real - mu_distorted
    diff_squared = diff.dot(diff)

    covmean, _ = sqrtm(sigma_real.dot(sigma_distorted), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = diff_squared + np.trace(sigma_real + sigma_distorted - 2 * covmean)
    return float(fad)
