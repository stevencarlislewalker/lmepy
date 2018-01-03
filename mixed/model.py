import numpy as np
import scipy.special
import functools
from scipy.sparse import coo_matrix


class MixedModel:
	def __init__(
		self,
		response_vector,
		fixed_effects_matrix,
		raw_random_effects_matrices,
		grouping_factor_indices
	):
		self._response_vector = response_vector  # type: np.ndarray
		self._fixed_effects_matrix = fixed_effects_matrix  # type: np.ndarray
		self._raw_random_effects_matrices = raw_random_effects_matrices  # type: dict
		self._grouping_factor_indices = grouping_factor_indices  # type: dict

	@property
	def response_vector(self):
		return self._response_vector

	@property
	def fixed_effects_matrix(self):
		return self._fixed_effects_matrix

	@property
	def raw_random_effects_matrices(self):
		return self._raw_random_effects_matrices

	@property
	def grouping_factor_indices(self):
		return self._grouping_factor_indices

	@property
	def number_samples(self):
		"""number of samples, n"""
		return self.response_vector.shape[0]

	@property
	def number_fixed_effect_columns(self):
		"""number of columns, p, of the fixed effects model matrix"""
		return self.fixed_effects_matrix.shape[1]

	@property
	def number_random_effect_terms(self):
		"""number of random effect terms"""
		return len(self.raw_random_effects_matrices)

	@property
	def number_raw_random_effects_columns_per_term(self):
		"""numbers of columns, p_i, in each raw random effects model matrix"""
		return {
			term: self.raw_random_effects_matrices[term].shape[1]
			for term
			in self.raw_random_effects_matrices
		}

	@property
	def number_grouping_factor_levels_per_term(self):
		"""numbers of levels, l_i, of the grouping factor indices for each term"""
		return {
			term: np.unique(self.grouping_factor_indices[term]).size
			for term
			in self.grouping_factor_indices
		}

	@property
	def number_random_effects_columns_per_term(self):
		"""numbers of columns, q_i, of the random effects model matrices per term"""
		return {
			term:
				self.number_raw_random_effects_columns_per_term[term] *
				self.number_grouping_factor_levels_per_term[term]
			for term
			in self.grouping_factor_indices
		}

	@property
	def number_random_effects_columns(self):
		"""number of columns, q, of the random effects model matrix"""
		return sum(self.number_random_effects_columns_per_term.values())

	@property
	def number_covariance_parameters_per_term(self):
		"""number of covariance parameters for each random effects term"""
		return {
			term: scipy.special.binom(
				self.number_raw_random_effects_columns_per_term[term] + 1,
				2
			)
			for term
			in self.number_raw_random_effects_columns_per_term
		}

	@property
	def number_covariance_parameters(self):
		"""number of covariance parameters"""
		return sum(self.number_covariance_parameters_per_term.values())

	@functools.lru_cache()
	def grouping_factor_matrices(self):
		return {
			term: coo_matrix(
				(
					np.repeat(
						[1],
						self.number_samples
					),
					(
						list(range(self.number_samples)),
						self.grouping_factor_indices[term]
					)
				),
				shape=(
					self.number_samples,
					self.number_grouping_factor_levels_per_term[term]
				)
			).tocsr()
			for term
			in self.grouping_factor_indices
		}
