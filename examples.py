import mixed
import numpy as np

mod = mixed.MixedModel(
	response_vector=np.array([2., 0.3, 4., -0.2, -3.2]),
	fixed_effects_matrix=np.array(
		[
			[1., 2.],
			[1., -2.],
			[1., 0.3],
			[1., -0.24],
			[1., -1.]
		]
	),
	raw_random_effects_matrices={
		'first': np.array(
			[
				[1.],
				[1.],
				[1.],
				[1.],
				[1.]
			]
		)
	},
	grouping_factor_indices={
		'first': np.array([0, 1, 1, 0, 0])
	}
)

mod.number_covariance_parameters
mod.number_covariance_parameters_per_term
mod.number_raw_random_effects_columns_per_term
mod.number_fixed_effect_columns
mod.number_grouping_factor_levels_per_term
mod.number_random_effect_terms
mod.number_random_effects_columns
mod.number_samples

mod.grouping_factor_matrices()

