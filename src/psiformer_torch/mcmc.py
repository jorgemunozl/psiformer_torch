"""
Implementation of Markov Chain Monte Carlo (MCMC) methods for sampling in PsiFormer.
Metropolis-Hastings algorithm is included for generating samples from complex distributions.
"""


class MCMC():
    """A class for generate samples from |psi|^2 (MCMC), with
    proposal distribution gaussian.
    """

    def __init__(self, mu, sigma):
        """
        Initialize the MCMC sampler with a Gaussian proposal distribution.

        Parameters:
        - mu: mean of the Gaussian proposal distribution.
        - sigma: standard deviation of the Gaussian proposal distribution.
        """
        self.mu = mu
        self.sigma = sigma
        self.batch_size = 1
        dtype = 'float32'



    def metropolis_hastings(target_distribution, proposal_distribution, initial_state, num_samples):
        """
        Perform Metropolis-Hastings sampling.

        Parameters:
        - target_distribution: function that computes the probability density of the target distribution.
        - proposal_distribution: function that generates a new sample given the current state.
        - initial_state: starting point for the Markov chain.
        - num_samples: number of samples to generate.

        Returns:
        - samples: list of generated samples.
        """
        samples = []
        current_state = initial_state

        for _ in range(num_samples):
            proposed_state = proposal_distribution(current_state)
            acceptance_ratio = (target_distribution(proposed_state) /
                                target_distribution(current_state))

            if acceptance_ratio >= 1 or random.uniform(0, 1) < acceptance_ratio:
                current_state = proposed_state

            samples.append(current_state)

        return samples
