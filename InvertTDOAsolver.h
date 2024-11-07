#ifndef INVERT_TDOA_SOLVER_H
#define INVERT_TDOA_SOLVER_H

#include "Eigen/Dense"
#include <array>

/**
 * @brief Function that finds the root for the invert TDOA problem.
 * 
 * Suppose \f$ x^1, x^2, x^3 \f$ represent the coordinates of the signal sources,
 * and \f$ y^1, y^2, y^3 \f$ represent the coordinates of the signal receivers.
 * Denote \f$ \delta^{ijk} := \| y^i - x^k \| - \| y^j - x^k \| \f$.
 * The pairwise distance difference matrix is defined as:
 * \f[
 * \begin{pmatrix}
 *     \delta^{121} & \delta^{131} & \delta^{231} \\
 *     \delta^{122} & \delta^{132} & \delta^{232} \\
 *     \delta^{123} & \delta^{133} & \delta^{233}
 * \end{pmatrix}
 * \f]
 *   
 * @param sourceCoordinates An array containing \f$ (x^1, x^2, x^3) \f$.
 * @param pddMatrix The pairwise distance difference matrix.
 * @return An array containing \f$ (y^1, y^2, y^3) \f$.
 * @throw runtime_error if a satisfactory root approximation could not be 
 * computed within 10 iterations.
 */
std::array<Eigen::Vector2d, 3> invertTDOAsolver(
	const std::array<Eigen::Vector2d, 3> &sourceCoordinates,
	const Eigen::Matrix3d& pddMatrix);

#endif //INVERT_TDOA_SOLVER_H